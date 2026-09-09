"""Evaluate one checkpoint across benchmarks. Run from the repo root:"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.core import checkpoints, report
from benchmarks.core.generate import T2I_KWARGS, image_path, read_completions, run_t2i, run_text, server_model, t2i_jobs
from benchmarks.core.registry import SPECS, BenchmarkSpec, load_items, load_metadata, load_prompts
from benchmarks.core.score import (
    GRADERS,
    RewardServiceClient,
    check_geneval2_metadata,
    score_images_local_geneval2,
)

_HELP = """Evaluate one checkpoint across benchmarks. Run from the repo root:

    python -m benchmarks.run --list
    python -m benchmarks.run -b image/geneval2 -b image/preference \\
        --ckpt stabilityai/stable-diffusion-3.5-medium --lora <unirl-ckpt-or-peft-dir> \\
        --reward-url http://<reward-host>:8080
    python -m benchmarks.run -b text/aime24 --endpoint http://127.0.0.1:30000
    python -m benchmarks.run --report

Stages: ``generate`` needs a GPU (t2i) or a serving endpoint (text); ``score``
needs the reward service (t2i) and is CPU-only for text. Results land in
``<out>/<ckpt-tag>/<benchmark>/`` (images/ or completions.jsonl, scores.jsonl,
summary.json); ``--report`` renders every summary under ``--out`` as markdown.
"""


def _parse_sim_even_batches(value: str) -> tuple[int, int]:
    try:
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        world, batch_size = (int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected WxB with positive integers, for example 32x32") from exc
    if world <= 0 or batch_size <= 0:
        raise argparse.ArgumentTypeError("W and B must both be positive integers")
    return world, batch_size


def _sim_even_order(num_items: int, world: int, batch_size: int) -> List[int]:
    """Indices after repeating the dataset prefix to fill complete W×B waves."""
    if num_items <= 0:
        return []
    wave_size = world * batch_size
    target_size = ((num_items + wave_size - 1) // wave_size) * wave_size
    return list(range(num_items)) + [index % num_items for index in range(target_size - num_items)]


def _sim_even_metrics(
    rows: List[Optional[Dict[str, float]]],
    keys: List[str],
    *,
    world: int,
    batch_size: int,
) -> Dict[str, float]:
    order = _sim_even_order(len(rows), world, batch_size)
    metrics: Dict[str, float] = {}
    for key in keys:
        values = [rows[index][key] for index in order if isinstance(rows[index], dict) and key in rows[index]]
        if values:
            metrics[f"{key}_sim{world}x{batch_size}"] = sum(values) / len(values)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-b", "--benchmark", action="append", default=[], help="benchmark name; repeatable (see --list)"
    )
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    parser.add_argument("--report", action="store_true", help="render all summaries under --out and exit")
    parser.add_argument("--ckpt", help="t2i base model: HF repo id or local path")
    parser.add_argument("--lora", help="PEFT adapter dir, or a UniRL checkpoint (auto-exported)")
    parser.add_argument("--endpoint", help="text: OpenAI-compatible server URL serving the checkpoint")
    parser.add_argument("--tag", help="results dir name (default: derived from --ckpt/--lora or the endpoint model)")
    parser.add_argument("--out", default="benchmarks_results", help="results root (default: %(default)s)")
    parser.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    parser.add_argument(
        "--reward-url",
        default=os.environ.get("REWARD_SERVICE_URL"),
        help="reward service base URL (or env REWARD_SERVICE_URL)",
    )
    parser.add_argument(
        "--num-prompts", type=int, help="truncate the prompt set (smoke runs; summary is flagged as subset)"
    )
    parser.add_argument("--samples-per-prompt", type=int, help="override the benchmark protocol k")
    parser.add_argument("--batch-size", type=int, default=4, help="t2i images per pipeline call")
    parser.add_argument("--seed", type=int, default=42, help="t2i base seed (image seed = seed + 1000*prompt + sample)")
    parser.add_argument("--steps", type=int, help="t2i: num_inference_steps (default: pipeline default)")
    parser.add_argument("--guidance", type=float, help="t2i: guidance_scale (default: pipeline default)")
    parser.add_argument("--height", type=int, help="t2i: image height (default: pipeline default)")
    parser.add_argument("--width", type=int, help="t2i: image width (default: pipeline default)")
    parser.add_argument("--shard", default="0/1", help="t2i generate: 'i/n' to split work across n processes")
    parser.add_argument(
        "--sim-even-batches",
        metavar="WxB",
        type=_parse_sim_even_batches,
        help="t2i: also report the metric under a simulated distributed eval that repeats "
        "the last partial wave (e.g. '32x32' = 32 GPUs x batch 32); default: 800 unique only",
    )
    parser.add_argument(
        "--local-geneval2",
        action="store_true",
        help="geneval2: score locally with transformers Qwen3-VL-8B (full-vocab softmax, geometric "
        "mean) instead of --reward-url; required to reproduce the DPPO paper numbers",
    )
    parser.add_argument(
        "--geneval2-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="local geneval2 scorer model id/path (used with --local-geneval2)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="text: concurrent requests to --endpoint")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    return parser.parse_args()


def _bench_dir(out: Path, tag: str, spec: BenchmarkSpec) -> Path:
    return out / tag / spec.name.replace("/", "_")


def _write_summary(
    bench_dir: Path,
    spec: BenchmarkSpec,
    tag: str,
    args,
    *,
    n_prompts: int,
    k: int,
    metrics: Dict,
    n_scored: int,
    n_errors: int,
) -> None:
    summary = {
        "benchmark": spec.name,
        "ckpt": tag,
        "base": args.ckpt,
        "adapter": args.lora,
        "n_prompts": n_prompts,
        "samples_per_prompt": k,
        "n_scored": n_scored,
        "n_errors": n_errors,
        "subset": bool(args.num_prompts),
        "metrics": metrics,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {spec.name}: " + "  ".join(f"{k_}={v:.4f}" for k_, v in metrics.items()))


def _run_t2i_benchmark(spec: BenchmarkSpec, tag: str, args, resolved: Optional[checkpoints.ResolvedCkpt]) -> None:
    prompts = load_prompts(spec)[: args.num_prompts or None]
    k = args.samples_per_prompt or spec.samples_per_prompt
    bench_dir = _bench_dir(Path(args.out), tag, spec)
    images_dir = bench_dir / "images"
    if args.dry_run:
        missing = len(t2i_jobs(prompts, images_dir, k))
        print(
            f"[plan] {spec.name}: {len(prompts)} prompts × {k} → {missing} images to generate; rewards={list(spec.rewards) or 'external'}"
        )
        return
    if args.stage in ("all", "generate"):
        if resolved is None:
            raise SystemExit(f"{spec.name}: --ckpt is required to generate images")
        gen_kwargs = dict(spec.gen)
        gen_kwargs.update({T2I_KWARGS[k_]: v for k_, v in vars(args).items() if k_ in T2I_KWARGS and v is not None})
        shard = tuple(int(x) for x in args.shard.split("/"))
        run_t2i(
            prompts,
            images_dir,
            resolved,
            samples_per_prompt=k,
            batch_size=args.batch_size,
            seed=args.seed,
            gen_kwargs=gen_kwargs,
            shard=shard,
            linspace_sigmas=spec.t2i_linspace_sigmas,
            prompt_seed=spec.t2i_prompt_seed,
        )
    if args.stage in ("all", "score"):
        if not spec.rewards:
            print(f"[score] {spec.name} is scored externally — see benchmarks/{spec.name}/README.md")
            return
        local_geneval2 = args.local_geneval2 and "geneval2" in spec.rewards
        if not local_geneval2 and not args.reward_url:
            raise SystemExit(f"{spec.name}: --reward-url (or REWARD_SERVICE_URL) is required to score")
        pairs, missing = [], 0
        for p in range(len(prompts)):
            for s in range(k):
                path = image_path(images_dir, p, s)
                if path.exists():
                    pairs.append((prompts[p], path))
                else:
                    missing += 1
        if missing:
            raise SystemExit(f"{spec.name}: {missing} images missing — finish --stage generate (all shards) first")
        metadatas = None
        if spec.send_metadata:
            per_prompt = load_metadata(spec)[: args.num_prompts or None]
            metadatas = [per_prompt[p] for p in range(len(prompts)) for _ in range(k)]
        if local_geneval2:
            rows, n_errors = score_images_local_geneval2(pairs, metadatas=metadatas, model_name=args.geneval2_model)
        else:
            client = RewardServiceClient(args.reward_url)
            client.check(spec.rewards)
            if "geneval2" in spec.rewards:
                check_geneval2_metadata(client)  # fail fast instead of silently scoring non-Soft-TIFA
            rows, n_errors = client.score_images(pairs, spec.rewards, metadatas=metadatas)
        with open(bench_dir / "scores.jsonl", "w") as f:
            for (prompt, path), row in zip(pairs, rows):
                f.write(json.dumps({"image": path.name, "prompt": prompt, "scores": row}) + "\n")
        scored = [row for row in rows if row]
        keys = sorted({k_ for row in scored for k_ in row})
        metrics = {
            k_: sum(row[k_] for row in scored if k_ in row) / max(1, sum(k_ in row for row in scored)) for k_ in keys
        }
        if args.sim_even_batches:
            world, batch_size = args.sim_even_batches
            metrics.update(_sim_even_metrics(rows, keys, world=world, batch_size=batch_size))
        _write_summary(
            bench_dir,
            spec,
            tag,
            args,
            n_prompts=len(prompts),
            k=k,
            metrics=metrics,
            n_scored=len(scored),
            n_errors=n_errors,
        )


def _run_text_benchmark(spec: BenchmarkSpec, tag: str, args) -> None:
    items = load_items(spec)[: args.num_prompts or None]
    k = args.samples_per_prompt or spec.samples_per_prompt
    bench_dir = _bench_dir(Path(args.out), tag, spec)
    completions_file = bench_dir / "completions.jsonl"
    if args.dry_run:
        print(
            f"[plan] {spec.name}: {len(items)} problems × {k} (avg@{k}), grader={spec.grader}, endpoint={args.endpoint or '<required>'}"
        )
        return
    if args.stage in ("all", "generate"):
        if not args.endpoint:
            raise SystemExit(f"{spec.name}: --endpoint is required to generate completions")
        run_text(
            items, completions_file, args.endpoint, samples_per_prompt=k, gen=spec.gen, concurrency=args.concurrency
        )
    if args.stage in ("all", "score"):
        if not completions_file.exists():
            raise SystemExit(f"{spec.name}: {completions_file} missing — run --stage generate (with --endpoint) first")
        responses: Dict[str, List[str]] = {}
        for row in read_completions(completions_file):
            responses.setdefault(row["id"], []).append(row["response"])
        n_short = sum(k - min(k, len(responses.get(item["id"], []))) for item in items)
        if n_short:
            raise SystemExit(f"{spec.name}: {n_short} completions missing — rerun --stage generate to fill them")
        grader = GRADERS[spec.grader]
        accs = []
        with open(bench_dir / "scores.jsonl", "w") as f:
            for item in items:
                acc = grader(item["answer"], responses[item["id"]][:k])
                accs.append(acc)
                f.write(json.dumps({"id": item["id"], "acc": acc, "n": k}) + "\n")
        metrics = {"acc": sum(accs) / len(accs)}
        _write_summary(
            bench_dir, spec, tag, args, n_prompts=len(items), k=k, metrics=metrics, n_scored=len(accs), n_errors=0
        )


def main() -> None:
    args = _parse_args()
    if args.list:
        for spec in SPECS.values():
            scoring = ",".join(spec.rewards) or spec.grader or "external"
            print(
                f"{spec.name:<20} {spec.modality:<5} k={spec.samples_per_prompt:<3} scoring={scoring:<30} {spec.notes}"
            )
        return
    if args.report:
        report.main(Path(args.out))
        return
    names = [n for arg in args.benchmark for n in arg.split(",")]
    if not names:
        raise SystemExit("pass -b <benchmark> (repeatable), --list, or --report")
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise SystemExit(f"unknown benchmarks {unknown}; see --list")

    t2i_specs = [SPECS[n] for n in names if SPECS[n].modality == "t2i"]
    resolved = None
    if t2i_specs and args.ckpt and not args.dry_run and args.stage in ("all", "generate"):
        resolved = checkpoints.resolve(args.ckpt, args.lora, Path(args.out))
    # Tag derivation must stay side-effect-free (no adapter export) for dry runs and score-only stages.
    if args.tag:
        tag = args.tag
    elif args.ckpt:
        tag = checkpoints.make_tag(args.ckpt, args.lora)
    elif args.endpoint and not args.dry_run:
        try:
            tag = checkpoints.make_tag(server_model(args.endpoint), None)
        except Exception as exc:  # noqa: BLE001 — endpoint down: tell the user, don't traceback
            raise SystemExit(
                f"cannot reach --endpoint {args.endpoint} ({exc}); pass --tag for score-only runs"
            ) from exc
    elif args.dry_run:
        tag = "dry-run"
    else:
        raise SystemExit("pass --tag (or --ckpt / --endpoint to derive one)")

    for name in names:
        spec = SPECS[name]
        if spec.modality == "t2i":
            _run_t2i_benchmark(spec, tag, args, resolved)
        else:
            _run_text_benchmark(spec, tag, args)


if __name__ == "__main__":
    main()
