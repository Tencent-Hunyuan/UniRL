"""Out-of-tree UniRL reward backend that scores H3 videos with the running framecode HPSv3 service (not part of the PR)."""

from __future__ import annotations

import time
from typing import List

import requests

from unirl.reward.base import RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse
from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient, jpeg_frames
from visual_rl.core.protocols.flash_hpsv3 import VideoContract


class FramecodeHPSv3Video(RewardBackend):
    """Video-only HPSv3, all frames, top-k mean, same wire protocol and JPEG path as framecode H3."""

    input_kind = "video"

    def __init__(self, *, origin: str, frames: int, height: int, width: int,
                 timeout_s: float = 1800.0, max_response_bytes: int = 8388608, batch_size: int = 8) -> None:
        super().__init__(model_name="hpsv3_video", batch_size=batch_size)
        self.contract = VideoContract(frames=frames, height=height, width=width)
        self.clients = []
        for item in origin.split(","):  # comma-separated identical instances; rows alternate between them
            identity = requests.get(item.rstrip("/") + "/healthz", timeout=30).json()["service_identity"]
            client = Hpsv3VideoClient(origin=item, service_identity=identity, contract=self.contract,
                                      timeout_s=timeout_s, max_response_bytes=max_response_bytes)
            client.healthcheck()
            self.clients.append(client)
        self.client = self.clients[0]

    def covers_prompt_video(self) -> bool:
        return True

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        prompts = self.prompts(request)
        ids = request.sample_ids or [f"row-{i}" for i in range(len(prompts))]
        rewards: List[float] = []
        videos = request.generated["video"].to_list()  # native [T, 3, H, W] views; skips request.videos' CTHW copies
        for row, (sample_id, prompt, video) in enumerate(zip(ids, prompts, videos)):
            frames = video.frames.float().cpu().contiguous()  # [T, 3, H, W] in [0, 1]
            score, _ = self.clients[row % len(self.clients)].score_video(sample_id=str(sample_id), prompt=prompt,
                                               frames_base64=jpeg_frames(frames, "BFCHW", self.contract))
            rewards.append(float(score))
        n = len(rewards)
        return RewardResponse(rewards=rewards, component_rewards={"hpsv3": list(rewards)},
                              successes=[True] * n, errors=[None] * n, compute_time=time.time() - start)
