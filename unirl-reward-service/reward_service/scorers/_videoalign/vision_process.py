"""Vision-info preprocessing — vendored from VideoAlign's
``vision_process.py`` (which itself derives from ``qwen_vl_utils``).

Reads videos via ``decord`` (preferred) or ``torchvision`` and packs
them into ``(T, C, H, W)`` tensors with smart frame-sampling and
spatial resizing. The chat-message format mirrors Qwen2-VL's vision
plugin: each video element is a dict with ``video`` (path / file://
URL), ``max_pixels``, ``fps`` *or* ``nframes``, and ``sample_type``.

Image / URL paths are kept for completeness but the VideoReward scorer
only goes through the video branch.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Resize so dims are divisible by ``factor`` and total pixels fall in
    ``[min_pixels, max_pixels]``, preserving aspect ratio."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = image_obj.convert("RGB")
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height, width, factor=size_factor, min_pixels=min_pixels, max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
    """Pick an even number of frames in ``[FRAME_FACTOR, total_frames]``."""
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if nframes > total_frames:
        nframes = total_frames
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def _read_video_torchvision(ele: dict) -> torch.Tensor:
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.debug(
        "torchvision read: video_path=%s total_frames=%d video_fps=%s elapsed=%.3fs",
        video_path, total_frames, video_fps, time.time() - st,
    )
    if ele["sample_type"] == "uniform":
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    elif ele["sample_type"] == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        fps = 8
        nframes = int(total_frames * fps // video_fps)
        frames_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes - frames_each_pts // 2 - 1)
        pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frames_idx[pt - frames_each_pts // 2 : pt + frames_each_pts // 2])
    else:
        raise ValueError(f"unknown sample_type: {ele['sample_type']!r}")
    video = video[idx]
    return video


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(ele: dict) -> torch.Tensor:
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    if "video_start" in ele or "video_end" in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.debug(
        "decord read: video_path=%s total_frames=%d video_fps=%s elapsed=%.3fs",
        video_path, total_frames, video_fps, time.time() - st,
    )
    if ele["sample_type"] == "uniform":
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    elif ele["sample_type"] == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        fps = 8
        nframes = int(total_frames * fps // video_fps)
        frames_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes - frames_each_pts // 2 - 1)
        pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frames_idx[pt - frames_each_pts // 2 : pt + frames_each_pts // 2])
    else:
        raise ValueError(f"unknown sample_type: {ele['sample_type']!r}")
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR):
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"], ele["resized_width"], factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height, width, factor=image_factor,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video

    assert isinstance(ele["video"], (list, tuple))
    process_info = ele.copy()
    process_info.pop("type", None)
    process_info.pop("video", None)
    images = [
        fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
        for video_element in ele["video"]
    ]
    nframes = ceil_by_factor(len(images), FRAME_FACTOR)
    if len(images) < nframes:
        images.extend([images[-1]] * (nframes - len(images)))
    return images


def extract_vision_info(conversations) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(conversations):
    """Read every image/video referenced by a chat-message list.

    Returns ``(image_inputs | None, video_inputs | None)`` —
    parallel to the order vision elements appear in ``conversations``.
    """
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
