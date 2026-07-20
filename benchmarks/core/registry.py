"""Benchmark registry: one :class:`BenchmarkSpec` per benchmark.

Adding a benchmark = drop its prompt data (or a fetch script) under
``benchmarks/<modality>/<name>/`` and register one spec here. All runner logic
lives in ``run.py`` / ``core/``; per-benchmark folders hold README + data only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

_MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
_MC_SUFFIX = "\nPlease reason step by step, and put the letter of your answer (A, B, C, or D) within \\boxed{}."


@dataclass(frozen=True)
class BenchmarkSpec:
    """Protocol card for one benchmark. ``data`` is repo-root-relative."""

    name: str
    modality: str  # "t2i": runner renders images | "text": runner queries an OpenAI-compatible endpoint
    data: str
    prompt_field: Optional[str] = None  # jsonl/csv/tsv column with the prompt; None for txt (one per line)
    samples_per_prompt: int = 4
    rewards: Tuple[str, ...] = ()  # reward-service scorer names; () = scored externally (see the README)
    send_metadata: bool = False  # t2i jsonl specs: ship each record as RewardRequest.metadata (geneval*)
    grader: Optional[str] = None  # text benchmarks: "math_verify" | "mc_letter"
    gen: Dict = field(default_factory=dict)  # generation defaults; CLI flags override
    notes: str = ""

    def data_path(self) -> Path:
        return REPO_ROOT / self.data

    def readme(self) -> str:
        """Repo-relative README that documents how to obtain this spec's data."""
        if self.data.startswith("benchmarks/"):
            return str(Path(self.data).parent.parent / "README.md")
        return f"benchmarks/{self.name}/README.md"


def load_prompts(spec: BenchmarkSpec) -> List[str]:
    """Unique prompts for a t2i benchmark, in file order."""
    path = spec.data_path()
    if not path.exists():
        raise SystemExit(f"{spec.name}: data file {path} missing — see {spec.readme()}")
    if path.suffix == ".jsonl":
        prompts = [json.loads(line)[spec.prompt_field] for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix in (".csv", ".tsv"):
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ",")
            prompts = [row[spec.prompt_field] for row in reader]
    else:  # txt: one prompt per line
        prompts = [line for line in path.read_text().splitlines() if line.strip()]
    return list(dict.fromkeys(prompts))


def load_metadata(spec: BenchmarkSpec) -> List[Dict]:
    """Full jsonl records aligned with :func:`load_prompts` order (first record wins
    per unique prompt). The reward-service geneval scorers read these as
    ``RewardRequest.metadata`` (``vqa_list`` / ``tag``+``include``)."""
    records: Dict[str, Dict] = {}
    for line in spec.data_path().read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records.setdefault(record[spec.prompt_field], record)
    return list(records.values())


def load_items(spec: BenchmarkSpec) -> List[Dict]:
    """Records for a text benchmark: standardized ``{id, problem, answer}`` jsonl."""
    path = spec.data_path()
    if not path.exists():
        raise SystemExit(f"{spec.name}: data file {path} missing — see {spec.readme()}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_ALL = (
    BenchmarkSpec(
        name="image/geneval2",
        modality="t2i",
        data="datasets/geneval2/synthetic/test.jsonl",
        prompt_field="prompt",
        rewards=("geneval2",),
        send_metadata=True,  # ships each record's vqa_list, so the scorer needs no dataset_path config
        notes="In-domain compositional T2I (VQAScore soft-TIFA via Qwen3-VL) — the set the GRPO/FlowDPPO recipes train on.",
    ),
    BenchmarkSpec(
        name="image/geneval",
        modality="t2i",
        data="benchmarks/image/geneval/data/evaluation_metadata.jsonl",
        prompt_field="prompt",
        rewards=("geneval",),
        send_metadata=True,  # ships tag/include/exclude, which the geneval scorer hard-requires
        notes="Official GenEval (Mask2Former + CLIP); enable the 'geneval' scorer in the reward service (off by default).",
    ),
    BenchmarkSpec(
        name="image/dpg_bench",
        modality="t2i",
        data="benchmarks/image/dpg_bench/data/dpg_bench.csv",
        prompt_field="text",
        notes="Dense-prompt following; scored externally with the official ELLA mPLUG script (see README).",
    ),
    BenchmarkSpec(
        name="image/preference",
        modality="t2i",
        data="benchmarks/image/preference/data/PartiPrompts.tsv",
        prompt_field="Prompt",
        samples_per_prompt=1,
        rewards=("hpsv3", "pickscore", "imagereward"),
        notes="Human-preference scores over PartiPrompts (P2).",
    ),
    BenchmarkSpec(
        name="text/aime24",
        modality="text",
        data="benchmarks/text/aime/data/aime2024.jsonl",
        samples_per_prompt=16,
        grader="math_verify",
        gen={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "prompt_suffix": _MATH_SUFFIX},
        notes="AIME 2024 (30 problems), avg@16.",
    ),
    BenchmarkSpec(
        name="text/aime25",
        modality="text",
        data="benchmarks/text/aime/data/aime2025.jsonl",
        samples_per_prompt=16,
        grader="math_verify",
        gen={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "prompt_suffix": _MATH_SUFFIX},
        notes="AIME 2025 (30 problems), avg@16.",
    ),
    BenchmarkSpec(
        name="text/math500",
        modality="text",
        data="benchmarks/text/math500/data/test.jsonl",
        samples_per_prompt=4,
        grader="math_verify",
        gen={"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384, "prompt_suffix": _MATH_SUFFIX},
        notes="MATH-500, avg@4.",
    ),
    BenchmarkSpec(
        name="text/gpqa",
        modality="text",
        data="benchmarks/text/gpqa/data/gpqa_diamond.jsonl",
        samples_per_prompt=4,
        grader="mc_letter",
        gen={"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384, "prompt_suffix": _MC_SUFFIX},
        notes="GPQA-Diamond (198 MC questions), avg@4. Gated data: run benchmarks/text/gpqa/fetch.py first.",
    ),
)

SPECS: Dict[str, BenchmarkSpec] = {spec.name: spec for spec in _ALL}
