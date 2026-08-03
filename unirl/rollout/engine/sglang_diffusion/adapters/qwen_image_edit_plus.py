"""Qwen-Image-Edit-Plus family: image-edit modality (text+image → image).

Sibling of :mod:`unirl.rollout.engine.sglang_diffusion.adapters.qwen_image`
(the T2I modality) with two image-edit deltas:

- **Request side.** Edit-Plus **requires** an aligned image conditioning Part
  (fail-fast if absent — Edit-Plus is edit-only). The adapter extracts PILs
  via :meth:`Images.to_pils` and injects each into the sampling kwargs under
  ``condition_image`` — a SamplingParams field injected by
  :mod:`._patches.patch_sampling_io` and copied onto ``Req.condition_image``
  in ``prepare_request``. SGLang's ``InputValidationStage`` checks
  ``batch.condition_image is not None`` BEFORE ``image_path``
  (input_validation.py:108), so the pre-populated PIL bypasses the file-path
  load entirely. Upstream's ``ImageVAEEncodingStage`` then VAE-encodes the
  source image and sets ``batch.image_latent`` (the packed
  ``[B, S_img, C*4]`` token-concat latent).
- **Response side.** The Edit-Plus-specific ``image_latent`` condition (the
  VAE-encoded source-image latent, needed by the trainer-side replay's
  token-concat in :meth:`QwenImageEditPlusDiffusionStep.predict_noise`) is
  captured by :mod:`._patches.patch_conditions` (which extends the IPC-
  survival machinery built for text embeds to also carry ``image_latent`` +
  ``image_latent_sizes`` off the batch). This adapter unpacks the packed
  latent to spatial ``[B, 16, H_img, W_img]`` and emits it as an
  :class:`ImageLatentCondition` alongside the inherited ``text`` /
  ``negative_text`` conditions.

Everything else (packed-trajectory unpack in ``build_segment``, CFG
semantics, LoRA wiring, weight sync) is inherited from the T2I adapter —
same checkpoint family, same text encoder, same noise-only trajectory (the
denoise loop concats ``batch.image_latent`` into a separate
``latent_model_input``, never back into ``latents``).

Upstream SGLang 0.5.12.post1 ships ``QwenImageEditPlusPipeline`` natively
(auto-selected via ``model_index.json`` ``_class_name``), so no pipeline
subclass is needed — unlike the vllm_omni path which subclasses
``RLQwenImageEditPlusPipeline``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image import QwenImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

_VAE_SCALE_FACTOR = 8


@register_adapter("qwen_image_edit_plus")
class QwenImageEditPlusAdapter(QwenImageAdapter):
    """Qwen-Image-Edit-Plus — text+image → image edit (single diffusion stage).

    Overrides only ``build_prompts`` (source-image ingestion) and
    ``build_condition`` (image_latent capture + unpack). The packed-trajectory
    unpack in :meth:`build_segment` is inherited unchanged — the Edit-Plus
    denoise loop records a noise-only trajectory (the image_latent concat lives
    in a separate ``latent_model_input`` per step, never written back to
    ``latents``), so the T2I unpack is correct.
    """

    pad_mask_to_embeds = True

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """Inject source-image PIL via ``condition_image`` sampling kwarg.

        Edit-Plus **requires** a source image per prompt (fail-fast if absent).
        The PIL is handed to SGLang verbatim — ``InputValidationStage``
        resizes it to condition_size + vae_size, ``ImageVAEEncodingStage``
        VAE-encodes it and sets ``batch.image_latent``. The driver never
        replicates VAE preprocessing.

        The K-expanded prompt collapse (``deexpand_prompts_from_groups``) is
        inherited from :meth:`ImageAdapter.build_prompts`; we only add the
        ``condition_image`` key alongside ``prompt``.
        """
        turns, image_batches = sample.vision_conditioning()
        text_turns = [turn.content for turn in turns if isinstance(turn.content, Texts)]
        if len(text_turns) != 1 or len(image_batches) != 1:
            raise ValueError(
                f"modality={self.model_family!r} requires exactly one text turn and one "
                f"image turn; got {len(text_turns)} text and {len(image_batches)} image turns."
            )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        prompts = list(text_turns[0].texts)
        unique_prompts, k = self._deexpand_prompts(prompts, gen_part.group_ids)
        images_prim = image_batches[0]
        pil_images = images_prim.to_pils()
        if len(pil_images) != len(prompts):
            raise ValueError(f"build_prompts: image batch {len(pil_images)} != prompt count {len(prompts)}")
        if len(prompts) != len(gen_part.sample_ids):
            raise ValueError(
                f"build_prompts: prompt count {len(prompts)} != diffusion sample count {len(gen_part.sample_ids)}"
            )
        if k > 1:
            unique_pils = self._first_per_group(pil_images, list(gen_part.group_ids))
            if len(unique_pils) != len(unique_prompts):
                raise ValueError(
                    f"build_prompts: collapsed image count {len(unique_pils)} != unique prompt "
                    f"count {len(unique_prompts)} (group_ids/image misalignment)."
                )
        else:
            unique_pils = pil_images
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": unique_pils if len(unique_pils) > 1 else unique_pils[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """T2I text-capture conditions + Edit-Plus ``image_latent``.

        Calls ``super().build_condition`` for the ``text`` / ``negative_text``
        slots, then collects the per-result ``image_latent`` (packed
        ``[1, S_img, C*4]``) + ``image_latent_sizes`` (the ``[(vae_width,
        vae_height)]`` pixel pair), unpacks each to spatial
        ``[1, 16, H_img, W_img]``, and emits the batched tensor as an
        :class:`ImageLatentCondition`.
        """
        cond_dict = super().build_condition(results)
        image_latents = self._collect_image_latents(results)
        if image_latents is not None:
            cond_dict["image_latent"] = ImageLatentCondition(latents=image_latents)
        return cond_dict

    def _collect_image_latents(self, results: List[RawResult]) -> Optional[torch.Tensor]:
        """Concatenate per-result image_latents, unpacked to spatial form.

        Each result's ``image_latent`` is the packed source-image latent
        ``[1, S_img, C*4]`` (one-element list injected by ``patch_conditions``).
        The paired ``image_latent_sizes`` carries ``[(vae_width, vae_height)]``
        (pixel W, H after upstream's ``calculate_vae_image_size`` resize to
        ~1024²). The latent grid is ``vae_height // vae_scale_factor`` ×
        ``vae_width // vae_scale_factor``.
        """
        from unirl.models.qwen_image.diffusion import _unpack_latents

        tensors: List[torch.Tensor] = []
        for r in results:
            packed_list = getattr(r, "image_latent", None)
            sizes_list = getattr(r, "image_latent_sizes", None)
            if not packed_list or not sizes_list:
                raise RuntimeError(
                    "build_condition: Qwen-Image-Edit-Plus rollout returned no "
                    "image_latent/image_latent_sizes. Check that patch_conditions "
                    "captured batch.image_latent (set by ImageVAEEncodingStage) "
                    "— the image_latent capture is required for trainer-side "
                    "replay (predict_noise concatenates it onto the noise latent)."
                )
            packed = packed_list[0]
            sizes = sizes_list[0]
            if len(sizes) != 1:
                raise NotImplementedError(
                    f"build_condition: multi-image Edit-Plus not supported (got {len(sizes)} source images per prompt)."
                )
            vae_width, vae_height = sizes[0]
            latent_h = int(vae_height) // _VAE_SCALE_FACTOR
            latent_w = int(vae_width) // _VAE_SCALE_FACTOR
            spatial = _unpack_latents(packed, latent_h=latent_h, latent_w=latent_w)
            tensors.append(spatial)
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            raise RuntimeError(
                f"build_condition: Qwen-Image-Edit-Plus image_latent tensors have "
                f"heterogeneous shapes {sorted(shapes)} — expected a uniform grid "
                f"(upstream normalizes to vae_size). Check that all source images "
                f"in the batch have the same aspect ratio, or extend the adapter "
                f"to ragged-pad."
            )
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _first_per_group(items: List[Any], group_ids: List[str]) -> List[Any]:
        """First item of each group, in first-seen group order.

        Mirrors how :func:`utils.deexpand_prompts_from_groups` collapses prompts,
        so the source-image collapse aligns with the prompt collapse regardless
        of the sample layout (contiguous or interleaved ``group_ids``).
        """
        seen: set[str] = set()
        out: List[Any] = []
        for item, gid in zip(items, group_ids):
            if gid not in seen:
                seen.add(gid)
                out.append(item)
        return out

    def _deexpand_prompts(self, prompts: List[str], group_ids: List[str]):
        """Collapse K-expanded prompts back to unique + repeat count.

        Thin wrapper around :func:`utils.deexpand_prompts_from_groups` so the
        import stays local (the base class imports utils at module level, but
        keeping the call explicit aids readability of the image-collapse
        parallel).
        """
        from unirl.rollout.engine.sglang_diffusion import utils

        return utils.deexpand_prompts_from_groups(prompts, list(group_ids))


__all__ = ["QwenImageEditPlusAdapter"]
