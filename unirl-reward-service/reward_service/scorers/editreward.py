"""EditReward scorer — multi-dimensional reward for instruction-guided image editing.

Wraps the EditRewardInferencer (Qwen2.5-VL-7B based) from the EditReward
package. The model evaluates instruction following and visual quality before
the configured head-pooling strategy produces the returned columns.

Column semantics depend on the pinned model config. With the official
EditReward-MiMo-VL-7B-SFT-2508 config (``pooling_strategy: mean``,
``output_dim: 2``, ``loss_type: uncertainty``) the two heads are pooled
INSIDE the model and the returned columns are ``[reward mean, log-sigma]``
— NOT two semantic scores. The sub-metric names below are historical wire
identifiers; under such configs ``edit_following`` carries the reward and
``edit_quality`` carries the heteroscedastic log-uncertainty, which is a
diagnostic, not a training signal. Consume the score channel only
(``sub_metric_reduce: first`` in recipes); never average the two.

Input convention:
    history[0] = (prompt, source_image)   — source image before editing
    history[1] = (prompt, edited_image)   — edited image after editing

The text prompt (editing instruction) is taken from history[0][0].
"""

from __future__ import annotations

import torch

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)


class EditRewardScorer(BaseScorer):
    """Multi-head edit reward scorer backed by EditRewardInferencer."""

    name = "editreward"
    version = "1"
    input_kind = "image"
    supports_offload = True
    sub_metric_names = ("edit_following", "edit_quality")

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        model_name_or_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rm_head_type: str = "ranknet_multi_head",
        offload_between_calls: bool = False,
    ) -> None:
        import os

        from reward_service.scorers._editreward import EditRewardInferencer

        self._target_device = device if torch.cuda.is_available() else "cpu"
        self._offload_between_calls = bool(offload_between_calls and self._target_device != "cpu")
        # UNIRL_SCORER_BOOT_OFFLOADED is the managed parent's per_call boot hint:
        # construct on CPU so the child never touches the GPU before it reports
        # ready (the server's boot offload then finds nothing to move). Unlike
        # offload_between_calls this does not change per-score behavior — the
        # parent drives onload/offload through the lifecycle endpoints.
        boot_offloaded = os.environ.get("UNIRL_SCORER_BOOT_OFFLOADED") == "1"
        initial_device = "cpu" if (self._offload_between_calls or boot_offloaded) else self._target_device

        # If checkpoint_path looks like a HF repo ID (not a local dir),
        # download it via huggingface_hub first.
        if not os.path.isdir(checkpoint_path) and "/" in checkpoint_path:
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(repo_id=checkpoint_path)

        self.inferencer = EditRewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            device=initial_device,
            reward_dim="dim1",
            rm_head_type=rm_head_type,
        )

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        try:
            if self._offload_between_calls:
                self.onload()
            return self._score_batch(items)
        finally:
            if self._offload_between_calls:
                self.offload()

    def _score_batch(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        # Score ALL valid items in ONE batched inferencer.reward() call. The
        # previous loop ran the 7B VLM once PER item (a batch of 1), so a
        # rollout's worth of scoring was N sequential forwards (~2h at 896
        # items). inferencer.reward() already takes parallel lists and runs a
        # single padded batch — the model was trained with batching, so batched
        # inference yields the same per-item scores at a fraction of the cost.
        results: list[dict[str, float]] = [None] * len(items)  # type: ignore[list-item]
        rows: list[int] = []
        prompts: list = []
        srcs: list = []
        edits: list = []
        for i, item in enumerate(items):
            try:
                prompt, source_image, edited_image = self._unpack_item(item)
                rows.append(i)
                prompts.append(prompt)
                srcs.append(source_image)
                edits.append(edited_image)
            except Exception:
                results[i] = {k: float("nan") for k in self.sub_metric_names}

        if rows:
            try:
                rewards = self.inferencer.reward(prompts=prompts, image_src=srcs, image_paths=edits)
                for row, i in enumerate(rows):
                    results[i] = self._shape_reward(rewards, row)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception:
                # A single bad item must not nan the whole batch — fall back to
                # the one-at-a-time path only on the error case.
                logger.exception("EditReward batch scoring failed; retrying items individually")
                for i in rows:
                    try:
                        results[i] = self._score_single(items[i])
                    except Exception:
                        results[i] = {k: float("nan") for k in self.sub_metric_names}

        return results

    def _shape_reward(self, rewards, row: int) -> dict[str, float]:
        """Map row ``row`` of a ``reward()`` output to the sub-metric dict.

        The single shaping path for both the batched forward and the
        one-at-a-time fallback (``_score_single`` calls this with row 0), keyed
        by the instance's ``sub_metric_names``.
        """
        first, second = self.sub_metric_names
        if isinstance(rewards, torch.Tensor):
            if rewards.dim() >= 2 and rewards.shape[-1] >= 2:
                return {
                    first: float(rewards[row, 0].item()),
                    second: float(rewards[row, 1].item()),
                }
            return {
                first: float(rewards[row].item()),
                second: float("nan"),
            }
        if isinstance(rewards, (list, tuple)):
            scores = []
            for reward in rewards:
                if torch.is_tensor(reward):
                    value = reward[row]
                    scores.append(float(value.reshape(-1)[0].item()))
                else:
                    scores.append(float(reward))
            return {
                first: scores[0] if scores else float("nan"),
                second: scores[1] if len(scores) > 1 else float("nan"),
            }
        raise TypeError(f"unsupported EditReward reward output: {type(rewards).__name__}")

    @staticmethod
    def _unpack_item(item: ScoreItem):
        if len(item.history) < 2:
            raise ValueError(f"EditReward requires 2 history turns (source + edited), got {len(item.history)}")
        prompt, source_image = item.history[0]
        _, edited_image = item.history[1]
        if source_image is None or edited_image is None:
            raise ValueError("Both source and edited images must be provided")
        return prompt, source_image, edited_image

    def _score_single(self, item: ScoreItem) -> dict[str, float]:
        """Score a single item.

        Expects item.history to have at least 2 turns:
            history[0] = (prompt, source_image)
            history[1] = (prompt, edited_image)
        """
        prompt, source_image, edited_image = self._unpack_item(item)

        # EditRewardInferencer.reward() accepts paths or PIL images
        # (process_vision_info handles PIL.Image directly)
        rewards = self.inferencer.reward(
            prompts=[prompt],
            image_src=[source_image],
            image_paths=[edited_image],
        )
        return self._shape_reward(rewards, 0)

    def onload(self) -> None:
        self.inferencer.model.to(self._target_device)
        self.inferencer.device = self._target_device

    def offload(self) -> None:
        self.inferencer.model.to("cpu")
        self.inferencer.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        if hasattr(self, "inferencer"):
            del self.inferencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register("editreward", EditRewardScorer)
