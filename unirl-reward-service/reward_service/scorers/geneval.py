"""GenEval compositional scorer via Mask2Former + CLIP color classification.

Evaluates compositional text-to-image generation using:
  - Object detection (Mask2Former via mmdetection)
  - Zero-shot color classification (open_clip + clip_benchmark)

Checks whether generated images satisfy compositional constraints:
  - Object count (single_object, two_object, counting)
  - Object color (colors, color_attr)
  - Relative position (position)

Constraints are passed via ``ScoreItem.metadata`` which must contain
``tag``, ``include`` (and optionally ``exclude``) fields. Example::

    metadata = {
        "tag": "two_object",
        "include": [
            {"class": "cat", "count": 1},
            {"class": "dog", "count": 1, "position": ["right of", 0]}
        ],
        "exclude": []
    }

Dependencies (heavy, isolated via envs/geneval.txt + Ray runtime_env):
  - mmdetection 2.x + mmcv-full (Mask2Former object detector)
  - open_clip + clip_benchmark (zero-shot color classification)

Reference: https://github.com/yifan123/reward-server/blob/main/reward_server/gen_eval.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]

# COCO class names used by Mask2Former for object detection.
_OBJECT_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "computer mouse", "tv remote", "computer keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# ── Helper functions ──────────────────────────────────────────────────────

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2, ...]."""
    def area_fn(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def _relative_position(
    obj_a: tuple, obj_b: tuple, position_threshold: float = 0.1
) -> set[str]:
    """Compute spatial position of object A relative to object B."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b

    revised_offset = (
        np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0)
        * np.sign(offset)
    )
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()

    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


class _ImageCrops(torch.utils.data.Dataset):
    """Dataset that yields cropped object regions for CLIP color classification."""

    def __init__(self, image: Image.Image, objects: list[tuple], transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999999")
        self._objects = objects
        self._transform = transform

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        box, mask = self._objects[index]
        if mask is not None:
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self._transform(image), 0)


# ── Scorer implementation ─────────────────────────────────────────────────

class GenEvalScorer(BaseScorer):
    """GenEval compositional reward scorer using Mask2Former + CLIP.

    Evaluates whether generated images satisfy compositional constraints
    (object count, color, position) specified in ScoreItem.metadata.
    """

    name = "geneval"
    sub_metric_names = ("geneval",)

    def __init__(
        self,
        mmdet_config: str = "",
        mmdet_checkpoint: str = "",
        clip_arch: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        score_type: str = "score",
        threshold: float = 0.3,
        counting_threshold: float = 0.9,
        max_objects: int = 16,
        nms_threshold: float = 1.0,
        position_threshold: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self.mmdet_config = mmdet_config
        self.mmdet_checkpoint = mmdet_checkpoint
        self.clip_arch = clip_arch
        self.clip_pretrained = clip_pretrained
        self.score_type = score_type  # "strict" | "score" | "reward"

        # Detection parameters
        self.threshold = threshold
        self.counting_threshold = counting_threshold
        self.max_objects = max_objects
        self.nms_threshold = nms_threshold
        self.position_threshold = position_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load models
        self._object_detector = None
        self._clip_model = None
        self._clip_transform = None
        self._clip_tokenizer = None
        self._color_classifiers: dict[str, Any] = {}

        self._load_object_detector()
        self._load_clip_model()
        logger.info(
            "GenEval scorer ready: detector=%s, clip=%s/%s, score_type=%s",
            Path(self.mmdet_checkpoint).name if self.mmdet_checkpoint else "none",
            self.clip_arch,
            self.clip_pretrained,
            self.score_type,
        )

    def _load_object_detector(self) -> None:
        """Initialize Mask2Former via mmdetection."""
        from mmdet.apis import init_detector

        if not self.mmdet_config:
            raise ValueError(
                "GenEval requires 'mmdet_config' path in params. "
                "Set it to the Mask2Former config .py file path."
            )
        if not self.mmdet_checkpoint:
            raise ValueError(
                "GenEval requires 'mmdet_checkpoint' path in params. "
                "Set it to the Mask2Former .pth checkpoint path."
            )

        logger.info(
            "Loading Mask2Former: config=%s, ckpt=%s, device=%s",
            self.mmdet_config, self.mmdet_checkpoint, self.device,
        )
        self._object_detector = init_detector(
            self.mmdet_config, self.mmdet_checkpoint, device=str(self.device)
        )

    def _load_clip_model(self) -> None:
        """Initialize open_clip model for color classification."""
        import open_clip

        logger.info("Loading CLIP: arch=%s, pretrained=%s", self.clip_arch, self.clip_pretrained)
        self._clip_model, _, self._clip_transform = open_clip.create_model_and_transforms(
            self.clip_arch, pretrained=self.clip_pretrained, device=str(self.device)
        )
        self._clip_tokenizer = open_clip.get_tokenizer(self.clip_arch)

    # ------------------------------------------------------------------
    # Color classification
    # ------------------------------------------------------------------

    def _get_color_classifier(self, classname: str):
        """Get or create a zero-shot color classifier for the given class."""
        if classname not in self._color_classifiers:
            from clip_benchmark.metrics import zeroshot_classification as zsc

            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self._clip_model,
                self._clip_tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                str(self.device),
            )
        return self._color_classifiers[classname]

    def _classify_colors(
        self, image: Image.Image, bboxes: list[tuple], classname: str
    ) -> list[str]:
        """Classify colors of detected objects using CLIP zero-shot."""
        from clip_benchmark.metrics import zeroshot_classification as zsc

        clf = self._get_color_classifier(classname)
        dataset = _ImageCrops(image, bboxes, self._clip_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, num_workers=0
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self._clip_model, clf, dataloader, str(self.device))
            return [COLORS[idx.item()] for idx in pred.argmax(1)]

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def _evaluate_strict(
        self, image: Image.Image, objects: dict[str, list[tuple]], metadata: dict
    ) -> tuple[bool, str]:
        """Strict binary evaluation: all constraints must be exactly satisfied."""
        correct = True
        reason = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[:req["count"]]

            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, "
                            f"found {colors.count(req['color'])} {req['color']}"
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if target_group < len(matched_groups) and matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = _relative_position(
                                    obj, target_obj, self.position_threshold
                                )
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, "
                                        f"found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break

            matched_groups.append(found_objects if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )

        return correct, "\n".join(reason)

    def _evaluate_reward(
        self, image: Image.Image, objects: dict[str, list[tuple]], metadata: dict
    ) -> tuple[bool, float, str]:
        """Continuous reward evaluation: penalizes count deviation proportionally."""
        correct = True
        reason = []
        rewards = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])

            # Count-based reward: 1 - |expected - found| / expected
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])

            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}=={req['count']}, found {len(found_objects)}"
                )
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found_objects, classname)
                    rewards.append(
                        1 - abs(req["count"] - colors.count(req["color"])) / req["count"]
                    )
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, "
                            f"found {colors.count(req['color'])} {req['color']}"
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        pos_correct = True
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = _relative_position(
                                    obj, target_obj, self.position_threshold
                                )
                                if expected_rel not in true_rels:
                                    correct = matched = pos_correct = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, "
                                        f"found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not pos_correct:
                                break
                        rewards.append(1.0 if pos_correct else 0.0)

            matched_groups.append(found_objects if matched else None)

        reward = sum(rewards) / len(rewards) if rewards else 0.0
        return correct, reward, "\n".join(reason)

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _detect_and_evaluate(
        self, image: Image.Image, metadata: dict
    ) -> dict[str, float]:
        """Run Mask2Former detection and evaluate a single image against its metadata."""
        from mmdet.apis import inference_detector

        np_image = np.array(image)
        result = inference_detector(self._object_detector, np_image)

        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(image)

        # Determine confidence threshold by task tag
        tag = metadata.get("tag", "")
        confidence_threshold = (
            self.counting_threshold if tag == "counting" else self.threshold
        )

        # Extract detected objects per class
        detected: dict[str, list[tuple]] = {}
        for index, classname in enumerate(_OBJECT_NAMES):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:self.max_objects].tolist()

            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((
                    bbox[index][max_obj],
                    None if segm is None else segm[index][max_obj],
                ))
                ordering = [
                    obj for obj in ordering
                    if self.nms_threshold == 1.0
                    or _compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.nms_threshold
                ]

            if not detected[classname]:
                del detected[classname]

        # Evaluate based on score_type
        if self.score_type == "strict":
            is_correct, _reason = self._evaluate_strict(image, detected, metadata)
            return {"geneval": 1.0 if is_correct else 0.0}
        elif self.score_type == "score":
            _is_correct, score, _reason = self._evaluate_reward(image, detected, metadata)
            return {"geneval": float(score)}
        else:  # "reward"
            is_correct, _reason = self._evaluate_strict(image, detected, metadata)
            return {"geneval": 1.0 if is_correct else 0.0}

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        """Score items based on compositional constraints in metadata.

        Each item's metadata must contain GenEval evaluation spec:
        - ``tag``: evaluation category (single_object, two_object, counting, etc.)
        - ``include``: list of required objects with class, count, and optionally color/position
        - ``exclude`` (optional): list of objects that should NOT appear
        """
        if not items:
            return []

        results: list[dict[str, float]] = []
        for i, item in enumerate(items):
            metadata = item.metadata
            if not metadata or "include" not in metadata:
                # A missing GenEval spec is a caller wiring bug (the rollout did
                # not thread per-prompt metadata through), not a per-item model
                # failure — so raise loudly instead of returning a plausible 0.0
                # that would silently train on a zero signal. The gateway turns
                # this into errors[i]["geneval"] for the request.
                raise ValueError(
                    f"geneval requires per-item metadata with 'tag'/'include' "
                    f"(item {i} has metadata={metadata!r}); wire the GenEval spec "
                    f"through RewardRequest.metadata."
                )

            _text, image = item.history[-1]
            result = self._detect_and_evaluate(image, metadata)
            results.append(result)

        return results

    def close(self) -> None:
        """Release model resources."""
        self._object_detector = None
        self._clip_model = None
        self._clip_transform = None
        self._clip_tokenizer = None
        self._color_classifiers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register("geneval", GenEvalScorer)
