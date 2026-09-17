"""DyRef SigLIPv2/CSD rewards with DRS and DAR metadata."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.reward.local.device import resolve_device
from unirl.types.primitives import Images, ImageSets
from unirl.types.reward import REWARD_MEDIA_METADATA_KEY, RewardRequest, RewardResponse
from unirl.utils.dtypes import parse_torch_dtype

from .base import LocalRewardBackend


def _target_image_uris(request: RewardRequest) -> List[str]:
    """Resolve one clean target-image URI from each metadata row."""
    if request.metadata is None or len(request.metadata) != request.batch_size:
        raise ValueError("DyRef rewards require one metadata row per generated sample.")
    targets: List[str] = []
    for row, metadata in enumerate(request.metadata):
        media = (metadata or {}).get(REWARD_MEDIA_METADATA_KEY)
        target_refs = [
            item
            for item in (media or [])
            if isinstance(item, dict) and item.get("modality") == "image" and item.get("role") == "target"
        ]
        if len(target_refs) != 1:
            raise ValueError(f"DyRef reward row {row} requires exactly one target image, got {len(target_refs)}.")
        uri = target_refs[0].get("uri")
        if not isinstance(uri, str) or uri.startswith(("http://", "https://", "s3://", "gs://")):
            raise NotImplementedError(f"DyRef reward row {row} requires a local target-image path, got {uri!r}.")
        targets.append(uri)
    return targets


def _reference_rows(request: RewardRequest) -> List[List[Image.Image]]:
    """Convert the single ordered condition-image turn to per-sample PIL rows."""
    if len(request.image_references) != 1:
        raise ValueError(
            "DyRef rewards require exactly one user image turn containing all ordered references; "
            f"got {len(request.image_references)}."
        )
    references = request.image_references[0]
    if isinstance(references, ImageSets):
        rows = references.to_pil_rows()
    elif isinstance(references, Images):
        rows = [[image] for image in references.to_pils()]
    else:
        raise TypeError(f"DyRef reward image turn must be Images or ImageSets, got {type(references).__name__}.")
    if len(rows) != request.batch_size or any(not row for row in rows):
        raise ValueError(
            "DyRef reward references must be non-empty and batch-aligned; "
            f"row_count={len(rows)}, batch={request.batch_size}, counts={[len(row) for row in rows]}."
        )
    return rows


def _dyref_metadata(request: RewardRequest, row: int) -> Dict[str, Any]:
    """Return one row's converter-authored DyRef metadata."""
    metadata = request.metadata[row] if request.metadata is not None else None
    value = (metadata or {}).get("dyref")
    if not isinstance(value, dict):
        raise ValueError(f"DyRef reward row {row} has no metadata.dyref mapping.")
    return value


class SigLIPSimilarityRewardScorer(LocalRewardBackend):
    """SigLIPv2 generated-to-target semantic similarity with optional DRS."""

    canonical_model_name = "siglip_similarity"

    def __init__(self, *, config: "SigLIPSimilaritySpec", base_device: str) -> None:
        dtype = parse_torch_dtype(config.dtype, field_name="SigLIPSimilaritySpec.dtype")
        device = resolve_device(config.device, base_device)
        if device == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=config.batch_size,
            model_id=config.model_id,
            feature_mode=config.feature_mode,
            apply_drs=config.apply_drs,
            drs_threshold=config.drs_threshold,
            drs_k1=config.drs_k1,
            drs_k2=config.drs_k2,
            drs_min_references=config.drs_min_references,
        )

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, Siglip2VisionModel

        model_id = self.model_kwargs["model_id"]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = Siglip2VisionModel.from_pretrained(model_id, torch_dtype=self.dtype).eval().to(self.device)

    def _features(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Extract normalized SigLIPv2 image features in configured batches."""
        features = []
        mode = str(self.model_kwargs["feature_mode"]).lower()
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(images=list(images[start : start + self.batch_size]), return_tensors="pt")
            inputs = {
                name: value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(self.device)
                for name, value in inputs.items()
            }
            outputs = self.model(**inputs)
            if mode == "flat":
                feature = outputs.last_hidden_state.flatten(1)
            elif mode in {"mean", "mean_pool"}:
                feature = outputs.last_hidden_state.mean(dim=1)
            elif mode in {"pooler", "pooler_output"} and getattr(outputs, "pooler_output", None) is not None:
                feature = outputs.pooler_output
            else:
                raise ValueError(f"SigLIPSimilaritySpec.feature_mode={mode!r}; expected flat, mean_pool, or pooler.")
            features.append(F.normalize(feature.float(), dim=-1))
        return torch.cat(features, dim=0)

    def score(self, request: RewardRequest) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw similarity, DRS reward, and reference counts."""
        generated = request.images or []
        if len(generated) != request.batch_size:
            raise ValueError(f"SigLIP similarity got {len(generated)} generated images for batch {request.batch_size}.")
        target_uris = _target_image_uris(request)
        references = _reference_rows(request)
        with torch.no_grad():
            generated_features = self._features(generated)
            unique_uris = list(dict.fromkeys(target_uris))
            unique_targets = []
            for uri in unique_uris:
                with Image.open(uri) as image:
                    unique_targets.append(image.convert("RGB"))
            unique_features = self._features(unique_targets)
            uri_to_index = {uri: index for index, uri in enumerate(unique_uris)}
            target_features = torch.stack([unique_features[uri_to_index[uri]] for uri in target_uris])
            raw = (generated_features * target_features).sum(dim=-1)

        counts = torch.tensor([len(row) for row in references], device=raw.device, dtype=raw.dtype)
        for row, count in enumerate(counts.tolist()):
            declared = int(_dyref_metadata(request, row).get("reference_count", -1))
            if declared != int(count):
                raise ValueError(
                    f"DyRef reward row {row} reference count mismatch: metadata={declared}, ImageSets={int(count)}."
                )
        if bool(self.model_kwargs["apply_drs"]):
            slope = float(self.model_kwargs["drs_k1"]) + float(self.model_kwargs["drs_k2"]) * (
                counts - float(self.model_kwargs["drs_min_references"])
            )
            shaped = torch.sigmoid(slope * (raw - float(self.model_kwargs["drs_threshold"])))
        else:
            shaped = raw
        return raw.cpu(), shaped.cpu(), counts.cpu()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        _, shaped, _ = self.score(request)
        return shaped.tolist()

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        raw, shaped, counts = self.score(request)
        return RewardResponse(
            rewards=shaped.tolist(),
            component_rewards={
                "semantic_raw": raw.tolist(),
                "semantic_drs": shaped.tolist(),
                "reference_count": counts.tolist(),
            },
            successes=[True] * request.batch_size,
            errors=[None] * request.batch_size,
            compute_time=time.time() - start,
        )


# Adapted from learn2phoenix/CSD; see CSD_LICENSE.
class _CSDCLIP(nn.Module):
    """Minimal official CSD ViT-L architecture needed for style inference."""

    def __init__(self, *, clip_download_root: Optional[str]) -> None:
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "CSD style reward requires OpenAI CLIP; install UniRL's `train` extra in the reward environment."
            ) from exc
        kwargs = {"download_root": clip_download_root} if clip_download_root else {}
        clip_model, _ = clip.load("ViT-L/14", device="cpu", jit=False, **kwargs)
        self.backbone = clip_model.visual.float()
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized CSD style descriptors."""
        features = self.backbone(inputs)
        return F.normalize(features @ self.last_layer_style, dim=1)


class CSDStyleRewardScorer(LocalRewardBackend):
    """CSD style similarity against the converter-marked style reference."""

    canonical_model_name = "csd_style"

    def __init__(self, *, config: "CSDStyleSpec", base_device: str) -> None:
        super().__init__(
            device=resolve_device(config.device, base_device),
            dtype=torch.float32,
            batch_size=config.batch_size,
            checkpoint_path=config.checkpoint_path,
            clip_download_root=config.clip_download_root,
        )

    def _load_model(self) -> None:
        checkpoint_path = Path(os.path.expanduser(self.model_kwargs["checkpoint_path"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CSD style checkpoint not found: {checkpoint_path}")
        model = _CSDCLIP(clip_download_root=self.model_kwargs.get("clip_download_root"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"CSD checkpoint {checkpoint_path} has no state dict.")
        state_dict = {
            (name.removeprefix("module.")): value
            for name, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        critical_missing = [
            name
            for name in incompatible.missing_keys
            if name == "last_layer_style" or name.startswith(("backbone.", "last_layer_style."))
        ]
        if critical_missing:
            raise ValueError(f"CSD checkpoint is missing required weights: {critical_missing[:5]}.")
        self.model = model.eval().to(self.device)

        import torchvision.transforms as transforms

        self.processor = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _features(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Extract normalized CSD style features in configured batches."""
        features = []
        for start in range(0, len(images), self.batch_size):
            pixels = torch.stack([self.processor(image) for image in images[start : start + self.batch_size]])
            features.append(self.model(pixels.to(self.device)).cpu())
        return torch.cat(features, dim=0)

    def score(self, request: RewardRequest) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CSD scores and a style-applicability mask."""
        generated = request.images or []
        references = _reference_rows(request)
        if len(generated) != request.batch_size:
            raise ValueError(f"CSD got {len(generated)} generated images for batch {request.batch_size}.")

        scores = torch.zeros(request.batch_size, dtype=torch.float32)
        applicable = torch.zeros(request.batch_size, dtype=torch.float32)
        selected_generated: List[Image.Image] = []
        selected_references: List[Image.Image] = []
        selected_keys: List[str] = []
        selected_rows: List[int] = []
        for row, reference_row in enumerate(references):
            metadata = _dyref_metadata(request, row)
            reference_types = metadata.get("reference_types")
            if not isinstance(reference_types, list) or len(reference_types) != len(reference_row):
                raise ValueError(
                    f"DyRef reward row {row} reference_types must align with {len(reference_row)} references."
                )
            style_positions = [index for index, kind in enumerate(reference_types) if kind == "style"]
            if len(style_positions) > 1:
                raise ValueError(f"DyRef reward row {row} has multiple style references at {style_positions}.")
            style_index = metadata.get("style_reference_index")
            if style_index is None:
                if style_positions:
                    raise ValueError(f"DyRef reward row {row} omits its style_reference_index.")
                continue
            style_index = int(style_index)
            if style_positions != [style_index]:
                raise ValueError(f"DyRef reward row {row} style index {style_index} disagrees with reference_types.")
            if not 0 <= style_index < len(reference_row):
                raise ValueError(
                    f"DyRef reward row {row} style_reference_index={style_index} "
                    f"is outside {len(reference_row)} references."
                )
            selected_rows.append(row)
            selected_generated.append(generated[row])
            selected_references.append(reference_row[style_index])
            selected_keys.append(str(metadata.get("index", row)))
            applicable[row] = 1.0

        if selected_rows:
            with torch.no_grad():
                generated_features = self._features(selected_generated)
                unique_keys = list(dict.fromkeys(selected_keys))
                first_reference = {key: selected_references[selected_keys.index(key)] for key in unique_keys}
                reference_features = self._features([first_reference[key] for key in unique_keys])
                key_to_index = {key: index for index, key in enumerate(unique_keys)}
                aligned_references = torch.stack([reference_features[key_to_index[key]] for key in selected_keys])
                scores[selected_rows] = (generated_features * aligned_references).sum(dim=-1)
        return scores, applicable

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        scores, _ = self.score(request)
        return scores.tolist()

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        scores, applicable = self.score(request)
        return RewardResponse(
            rewards=scores.tolist(),
            component_rewards={"style_applicable": applicable.tolist()},
            successes=[True] * request.batch_size,
            errors=[None] * request.batch_size,
            compute_time=time.time() - start,
        )


class DyRefRewardScorer(RewardBackend):
    """Compose SigLIPv2 DRS and CSD while emitting DAR group weights."""

    def __init__(self, *, config: "DyRefRewardSpec", base_device: str) -> None:
        super().__init__(model_name="dyref", batch_size=config.semantic.batch_size)
        if config.semantic_weight <= 0.0 or config.style_weight < 0.0:
            raise ValueError("DyRef reward weights require semantic_weight > 0 and style_weight >= 0.")
        if config.dar_gamma <= 0.0 or not 0.0 < config.dar_min_weight <= config.dar_max_weight:
            raise ValueError("DyRef DAR requires gamma > 0 and 0 < min_weight <= max_weight.")
        self.semantic_weight = float(config.semantic_weight)
        self.style_weight = float(config.style_weight)
        self.dar_gamma = float(config.dar_gamma)
        self.dar_min_weight = float(config.dar_min_weight)
        self.dar_max_weight = float(config.dar_max_weight)
        self.dar_eps = float(config.dar_eps)
        self.semantic = SigLIPSimilarityRewardScorer(config=config.semantic, base_device=base_device)
        self.style = (
            CSDStyleRewardScorer(config=config.style, base_device=base_device) if self.style_weight > 0.0 else None
        )

    def _dar_weights(self, raw: torch.Tensor, group_ids: Optional[List[str]]) -> torch.Tensor:
        """Compute normalized paper-equation DAR weights per complete prompt group."""
        if group_ids is None or len(group_ids) != raw.numel():
            raise ValueError("DyRef DAR requires one group_id per reward row.")
        unique_groups = list(dict.fromkeys(group_ids))
        weights: Dict[str, float] = {}
        for group_id in unique_groups:
            indices = [index for index, value in enumerate(group_ids) if value == group_id]
            mean_reward = float(raw[indices].clamp(0.0, 1.0).mean().item())
            weight = (max(0.0, 1.0 - mean_reward) + self.dar_eps) ** self.dar_gamma
            weights[group_id] = min(self.dar_max_weight, max(self.dar_min_weight, weight))
        mean_weight = sum(weights.values()) / len(weights)
        normalized = {
            group_id: min(
                self.dar_max_weight,
                max(self.dar_min_weight, weight / max(mean_weight, self.dar_eps)),
            )
            for group_id, weight in weights.items()
        }
        return torch.tensor([normalized[group_id] for group_id in group_ids], dtype=torch.float32)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        try:
            semantic_raw, semantic_drs, reference_count = self.semantic.score(request)
            if self.style is None:
                style = torch.zeros_like(semantic_drs)
                style_applicable = torch.zeros_like(semantic_drs)
            else:
                style, style_applicable = self.style.score(request)
            rewards = self.semantic_weight * semantic_drs + self.style_weight * style
            dar_weight = self._dar_weights(semantic_raw, request.group_ids)
            return RewardResponse(
                rewards=rewards.tolist(),
                component_rewards={
                    "semantic_raw": semantic_raw.tolist(),
                    "semantic_drs": semantic_drs.tolist(),
                    "style_csd": style.tolist(),
                    "style_applicable": style_applicable.tolist(),
                    "reference_count": reference_count.tolist(),
                    "dar_weight": dar_weight.tolist(),
                },
                successes=[True] * request.batch_size,
                errors=[None] * request.batch_size,
                compute_time=time.time() - start,
            )
        except Exception as exc:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(exc)] * request.batch_size,
                compute_time=time.time() - start,
            )

    def is_available(self) -> bool:
        return self.semantic.is_available() and (self.style is None or self.style.is_available())

    def offload(self) -> None:
        self.semantic.offload()
        if self.style is not None:
            self.style.offload()

    def onload(self) -> None:
        self.semantic.onload()
        if self.style is not None:
            self.style.onload()


@dataclass
class SigLIPSimilaritySpec(BaseRewardComponentSpec):
    """Typed config for DyRef semantic similarity and DRS."""

    batch_size: int = 8
    device: str = "auto"
    dtype: Any = "bf16"
    model_id: str = "google/siglip2-base-patch16-384"
    feature_mode: str = "flat"
    apply_drs: bool = True
    drs_threshold: float = 0.65
    drs_k1: float = 10.0
    drs_k2: float = 3.0
    drs_min_references: int = 2


@dataclass
class CSDStyleSpec(BaseRewardComponentSpec):
    """Typed config for the official CSD ViT-L style checkpoint."""

    batch_size: int = 8
    device: str = "auto"
    checkpoint_path: str = ""
    clip_download_root: Optional[str] = None


@dataclass
class DyRefRewardSpec(BaseRewardComponentSpec):
    """Typed config for the complete DyRef reward stack."""

    semantic: SigLIPSimilaritySpec = field(default_factory=SigLIPSimilaritySpec)
    style: CSDStyleSpec = field(default_factory=CSDStyleSpec)
    semantic_weight: float = 1.0
    style_weight: float = 0.8
    dar_gamma: float = 2.0
    dar_min_weight: float = 0.01
    dar_max_weight: float = 5.0
    dar_eps: float = 1e-6


__all__ = [
    "CSDStyleRewardScorer",
    "CSDStyleSpec",
    "DyRefRewardScorer",
    "DyRefRewardSpec",
    "SigLIPSimilarityRewardScorer",
    "SigLIPSimilaritySpec",
]
