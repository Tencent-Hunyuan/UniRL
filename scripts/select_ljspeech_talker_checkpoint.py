"""Compare UniRL Talker LoRA checkpoints on fixed sampled WER/EOS metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from unirl.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from unirl.models.qwen3_omni.talker_bundle import Qwen3OmniTalkerBundle
from unirl.models.qwen3_omni.talker_pipeline import Qwen3OmniTalkerPipeline
from unirl.reward.local.tts_metrics import score_edit_metric
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.utils.eval_talker_tts_baseline import _load_unirl_lora_checkpoint


def _rows(path: str, count: int):
    result = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
            if len(result) == count:
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--asr_model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    candidates = sorted(
        Path(args.checkpoint_root).glob("checkpoint-*/checkpoint.pt"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-*/checkpoint.pt under {args.checkpoint_root}")
    rows = _rows(args.eval_jsonl, args.samples)

    config = Qwen3OmniPipelineConfig(
        pretrained_model_ckpt_path=args.model_path,
        model_precision="bf16",
        device=args.device,
        enable_talker=True,
    )
    bundle = Qwen3OmniTalkerBundle.from_config(config)
    _load_unirl_lora_checkpoint(bundle, str(candidates[0]))
    pipeline = Qwen3OmniTalkerPipeline.from_bundle(bundle, decode_audio=True)
    codec_eos = int(bundle.config.talker_config.codec_eos_token_id)

    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=0 if args.device.startswith("cuda") else -1,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    )

    summaries = []
    for checkpoint_path in candidates:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = bundle.talker.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"{checkpoint_path}: unexpected keys {incompatible.unexpected_keys[:8]}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        scored = []
        for row in rows:
            metadata = row["metadata"]
            transcript = str(metadata["normalized_transcript"])
            sample_id = str(row["sample_id"])
            request = Sample(
                parts=[
                    Part.input(
                        [sample_id],
                        primitives={"text": Texts(texts=[transcript])},
                        metadata=[{"speaker": str(metadata["speaker"])}],
                    )
                ]
            ).fork(
                1,
                sampling_params=ARSamplingParams(
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.9,
                    top_p=1.0,
                    top_k=50,
                ),
            )
            output = pipeline.generate(request).parts[-1]
            tokens = output.segment.tokens
            eos_ok = bool(tokens.numel() and int(tokens[-1]) == codec_eos)
            waveform = output.primitives["audio"].to_list()[0].waveform
            prediction = asr(
                {
                    "array": waveform.detach().float().cpu().numpy(),
                    "sampling_rate": 24000,
                },
                generate_kwargs={"language": "english"},
            )
            hypothesis = str(prediction.get("text", "")).strip()
            wer = float(score_edit_metric(transcript, hypothesis, metric="wer", language="en")["rate"])
            scored.append(
                {
                    "sample_id": sample_id,
                    "wer": wer,
                    "eos_ok": eos_ok,
                    "codec_steps": int(tokens.numel()),
                    "hypothesis": hypothesis,
                }
            )
        summary = {
            "step": int(checkpoint.get("step", -1)),
            "checkpoint": str(checkpoint_path),
            "mean_wer": sum(item["wer"] for item in scored) / len(scored),
            "eos_failure_rate": sum(not item["eos_ok"] for item in scored) / len(scored),
            "rows": scored,
        }
        summaries.append(summary)
        print(json.dumps({key: summary[key] for key in ("step", "mean_wer", "eos_failure_rate")}), flush=True)

    result = {
        "samples": len(rows),
        "seed": args.seed,
        "best_by_wer": min(summaries, key=lambda item: item["mean_wer"])["step"],
        "checkpoints": summaries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
