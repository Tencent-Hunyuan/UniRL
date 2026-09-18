"""Prepare interleaved-image agent SFT manifests from the SearchGen-20K traces — see README.md."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sqlite3
import tarfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from unirl.data.sft import normalize_supervised_example

SYSTEM_PROMPT = (
    "You are an image-generation planning agent. Given a user's image request you: "
    "(1) analyze the prompt and issue image search queries for missing visual knowledge; "
    "(2) after seeing the retrieved candidate images, select the references worth keeping; "
    "(3) write the final refined generation prompt, citing any borrowed visual elements as "
    "'Reference Image N'."
)
REFINE_INSTRUCTION = (
    "Write the final refined generation prompt. When borrowing a visual element, cite its source as "
    "'Reference Image N', numbered in the order selected above. End with the line "
    "'Selected references: [..]' listing the references selected in the previous step."
)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON — {exc}") from exc


def _full_trace(trace: Dict[str, Any]) -> bool:
    av = trace.get("artifact_availability") or {}
    required = (
        "s1_analysis",
        "s1d1_search_results",
        "s2_reference_selection",
        "s3_refined_prompt",
        "s4_generation_manifest",
    )
    return all(av.get(k) for k in required)


class CandidateIndex:
    """Read-only view of the corpus search DB: query_id → ranked downloaded candidates."""

    def __init__(self, corpus_root: str) -> None:
        db_path = os.path.join(corpus_root, "metadata", "search.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"searchgen prepare_sft: corpus database not found at {db_path}")
        self._db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._cache: Dict[str, List[Tuple[str, str, str]]] = {}

    def candidates(self, query_id: str) -> List[Tuple[str, str, str]]:
        """Ranked ``(entry_id, title, portable_path)`` rows with a downloaded asset."""
        if query_id not in self._cache:
            rows = self._db.execute(
                """SELECT qe.entry_id, COALESCE(qe.observed_title, e.representative_title, ''), a.portable_path
                   FROM query_entries qe
                   JOIN entries e USING(entry_id)
                   JOIN assets a ON a.asset_id = e.asset_id
                   WHERE qe.query_id = ? AND e.download_status = 'downloaded'
                     AND e.search_type = 'image'
                   ORDER BY qe.source_rank""",
                (query_id,),
            ).fetchall()
            self._cache[query_id] = [(str(e), str(t), str(p)) for e, t, p in rows]
        return self._cache[query_id]


def build_trace_conversation(
    trace: Dict[str, Any],
    *,
    user_prompt: str,
    index: CandidateIndex,
    candidates_per_query: int,
) -> Optional[Tuple[List[Dict[str, Any]], List[str]]]:
    """One trace → (6-turn conversation with image URIs as portable paths, used paths)."""
    queries = [q for q in trace.get("s1d1_search_queries", []) if q.get("query_id")]
    if not queries:
        return None
    # "Reference Image N" numbering follows s4 ordered_references (see README.md);
    # any unresolved or reordered entry would shift it, so drop the trace whole.
    selections = (trace.get("s4_generation_manifest") or {}).get("ordered_references") or []
    if not selections:
        return None
    if any(
        not s.get("query_id") or not s.get("entry_id") or s.get("unresolved_reason") or s.get("index") != i
        for i, s in enumerate(selections)
    ):
        return None

    selected_by_query: Dict[str, set] = defaultdict(set)
    for sel in selections:
        selected_by_query[sel["query_id"]].add(sel["entry_id"])

    # Per query: top-N downloaded candidates, forced to include that query's selections.
    result_parts: List[Dict[str, Any]] = [{"type": "text", "text": "Search results:"}]
    position_by_entry: Dict[Tuple[str, str], int] = {}
    used_paths: List[str] = []
    image_no = 0
    for qi, q in enumerate(queries):
        rows = index.candidates(q["query_id"])
        keep = rows[:candidates_per_query]
        kept_ids = {e for e, _, _ in keep}
        keep += [row for row in rows[candidates_per_query:] if row[0] in selected_by_query[q["query_id"]] - kept_ids]
        header = f"Query {qi + 1}: {q.get('query', '')}"
        if not keep:
            result_parts.append({"type": "text", "text": f"{header}\n(no image results)"})
            continue
        result_parts.append({"type": "text", "text": header})
        for entry_id, title, path in keep:
            image_no += 1
            position_by_entry.setdefault((q["query_id"], entry_id), image_no)
            label = f"Image {image_no}: {title}" if title else f"Image {image_no}:"
            result_parts.append({"type": "text", "text": label})
            result_parts.append({"type": "image", "image": path})
            used_paths.append(path)

    # Every real selection must be present in the shown candidates. The selection
    # target welds the two numbering spaces together explicitly: candidates are
    # "Image <global position>", kept references are "Reference Image <selection
    # order>" (the numbering the refined prompt cites verbatim).
    selection_lines: List[str] = []
    for ref_no, sel in enumerate(selections, start=1):
        pos = position_by_entry.get((sel["query_id"], sel["entry_id"]))
        if pos is None:
            return None
        reasoning = str(sel.get("selection_reasoning") or "").strip()
        line = f"Selected Image {pos} as Reference Image {ref_no}"
        selection_lines.append(f"{line}: {reasoning}" if reasoning else line)

    analysis = trace.get("s1_prompt_analysis") or {}
    analysis_text = str(analysis.get("analysis_reasoning") or "").strip()
    query_lines = [f"{qi + 1}. {q.get('query', '')}" for qi, q in enumerate(queries)]
    s1_text = (analysis_text + "\n\n" if analysis_text else "") + "Search queries:\n" + "\n".join(query_lines)

    s3 = trace.get("s3_refined_prompt") or {}
    refined = str(s3.get("refined_prompt") or "").strip()
    ref_indices = s3.get("selected_reference_indices") or []
    if not refined or not isinstance(ref_indices, list) or not ref_indices:
        return None
    if max(int(i) for i in ref_indices) > len(selections) or min(int(i) for i in ref_indices) < 1:
        return None
    s3_text = f"{refined}\n\nSelected references: {[int(i) for i in ref_indices]}"

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": s1_text},
        {"role": "user", "content": result_parts},
        {"role": "assistant", "content": "\n".join(selection_lines)},
        {"role": "user", "content": REFINE_INSTRUCTION},
        {"role": "assistant", "content": s3_text},
    ]
    return conversation, used_paths


def expand_trace(
    trace_id: str, conversation: List[Dict[str, Any]], *, metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """One linearized trace → one record per supervised assistant turn (s1/s2/s3)."""
    records = []
    stage_names = iter(("s1_analysis", "s2_selection", "s3_refine"))
    for turn, message in enumerate(conversation):
        if message["role"] != "assistant":
            continue
        stage = next(stage_names)
        records.append(
            {
                "sample_id": f"{trace_id}:{stage.split('_')[0]}",
                "messages": conversation[: turn + 1],
                "metadata": {**metadata, "stage": stage},
            }
        )
    return records


class ShardExtractor:
    """Extract corpus TAR members → downscaled RGB JPEGs under ``<out>/images``."""

    def __init__(self, corpus_root: str, out_images_dir: str, *, max_image_px: int, workers: int = 8) -> None:
        self.corpus_root = corpus_root
        self.out_dir = out_images_dir
        self.max_px = int(max_image_px)
        self.workers = max(1, int(workers))
        self._member_to_shard: Optional[Dict[str, str]] = None

    def _load_index(self) -> Dict[str, str]:
        if self._member_to_shard is None:
            index_path = os.path.join(self.corpus_root, "data", "cached-images", "shards.jsonl")
            self._member_to_shard = {}
            for row in _read_jsonl(index_path):
                self._member_to_shard[row["member_name"]] = row["shard_path"]
        return self._member_to_shard

    def out_name(self, portable_path: str) -> str:
        base = os.path.splitext(os.path.basename(portable_path))[0]
        return f"images/{base}.jpg"

    def _extract_shard(self, shard: str, members: List[str]) -> Dict[str, str]:
        """Stream one shard, materializing the wanted members; returns its done map."""
        from PIL import Image as PILImage

        done: Dict[str, str] = {}
        wanted = set(members)
        shard_abs = os.path.join(self.corpus_root, shard)
        if not os.path.exists(shard_abs):
            print(f"warning: shard {shard} missing from the corpus mirror — dropping {len(wanted)} member(s)")
            return done
        with tarfile.open(shard_abs) as tf:
            while wanted:
                member = tf.next()
                if member is None:
                    break
                if member.name not in wanted:
                    continue
                wanted.discard(member.name)
                payload = tf.extractfile(member)
                if payload is None:
                    continue
                try:
                    img = PILImage.open(io.BytesIO(payload.read()))
                    img.load()
                except Exception:
                    continue
                img = img.convert("RGB")
                if max(img.size) > self.max_px:
                    scale = self.max_px / max(img.size)
                    img = img.resize(
                        (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))),
                        PILImage.BICUBIC,
                    )
                uri = self.out_name(member.name)
                target = os.path.join(self.out_dir, os.path.basename(uri))
                tmp = f"{target}.tmp.{os.getpid()}"
                img.save(tmp, "JPEG", quality=90)
                os.replace(tmp, target)
                done[member.name] = uri
        if wanted:
            print(f"warning: {len(wanted)} member(s) not found in {shard}")
        return done

    def extract(self, portable_paths: Sequence[str]) -> Dict[str, str]:
        """Materialize the given members; returns {portable_path: manifest-relative URI}."""
        from concurrent.futures import ThreadPoolExecutor

        os.makedirs(self.out_dir, exist_ok=True)
        done: Dict[str, str] = {}
        pending: Dict[str, List[str]] = defaultdict(list)
        index = self._load_index()
        unindexed = 0
        for path in dict.fromkeys(portable_paths):
            uri = self.out_name(path)
            if os.path.exists(os.path.join(self.out_dir, os.path.basename(uri))):
                done[path] = uri
                continue
            shard = index.get(path)
            if shard is None:
                unindexed += 1
                continue
            pending[shard].append(path)
        if unindexed:
            print(f"warning: {unindexed} member(s) absent from the shard index — their traces will be dropped")

        if pending:
            print(f"extracting {sum(len(m) for m in pending.values())} image(s) from {len(pending)} shard(s)...")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for shard_done in pool.map(lambda kv: self._extract_shard(*kv), sorted(pending.items())):
                    done.update(shard_done)
        return done


def _rewrite_image_uris(conversation: List[Dict[str, Any]], mapping: Dict[str, str]) -> bool:
    """Swap portable paths for extracted URIs in-place; False if any image is missing."""
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image":
                continue
            uri = mapping.get(part["image"])
            if uri is None:
                return False
            part["image"] = uri
    return True


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--searchgen-root", required=True, help="dir containing searchgen-20k/ and searchgen-corpus-1m/"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-traces", type=int, default=0, help="cap on rendered traces (0 = all)")
    parser.add_argument("--candidates-per-query", type=int, default=4)
    parser.add_argument(
        "--max-image-px",
        type=int,
        default=448,
        help="long-side cap for extracted candidate images (the budget the agent-SFT recipes assume); "
        "changing it does not re-extract existing images — use a fresh --out-dir",
    )
    parser.add_argument("--extract-workers", type=int, default=8, help="parallel shard readers for image extraction")
    parser.add_argument("--max-target-chars", type=int, default=8000, help="drop records with longer target turns")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1.")
    if args.candidates_per_query < 1:
        raise SystemExit("--candidates-per-query must be at least 1.")

    root = os.path.abspath(args.searchgen_root)
    sg20k = os.path.join(root, "searchgen-20k")
    corpus = os.path.join(root, "searchgen-corpus-1m")
    prompts = {
        row["row_id"]: str(row.get("user_prompt") or "")
        for row in _read_jsonl(os.path.join(sg20k, "metadata", "train_metadata.jsonl"))
    }
    index = CandidateIndex(corpus)

    conversations: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], List[str]]] = []
    dropped = {"partial": 0, "unrenderable": 0}
    for trace in _read_jsonl(os.path.join(sg20k, "traces", "trace_artifacts.jsonl")):
        if not _full_trace(trace):
            dropped["partial"] += 1
            continue
        user_prompt = prompts.get(trace.get("row_id"), "")
        if not user_prompt:
            dropped["unrenderable"] += 1
            continue
        built = build_trace_conversation(
            trace,
            user_prompt=user_prompt,
            index=index,
            candidates_per_query=args.candidates_per_query,
        )
        if built is None:
            dropped["unrenderable"] += 1
            continue
        meta = {
            "row_id": trace["row_id"],
            "trace_id": trace["trace_id"],
            "reasoner_id": trace.get("reasoner_id"),
        }
        conversations.append((trace["trace_id"], meta, built[0], built[1]))
        if args.max_traces > 0 and len(conversations) >= args.max_traces:
            break
    if len(conversations) < 2:
        raise SystemExit(f"searchgen prepare_sft: only {len(conversations)} renderable traces found.")

    extractor = ShardExtractor(
        corpus,
        os.path.join(args.out_dir, "images"),
        max_image_px=args.max_image_px,
        workers=args.extract_workers,
    )
    mapping = extractor.extract([p for _, _, _, paths in conversations for p in paths])

    by_row: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_image_missing = n_overlong = 0
    for trace_id, meta, conversation, _paths in conversations:
        if not _rewrite_image_uris(conversation, mapping):
            n_image_missing += 1
            continue
        for record in expand_trace(trace_id, conversation, metadata=meta):
            if len(str(record["messages"][-1]["content"])) > args.max_target_chars:
                n_overlong += 1
                continue
            normalize_supervised_example(record, default_sample_id=record["sample_id"], base_dir=args.out_dir)
            by_row[meta["row_id"]].append(record)

    row_ids = sorted(by_row)
    if len(row_ids) < 2:
        raise SystemExit("searchgen prepare_sft: fewer than two source rows survived filtering.")
    random.Random(args.seed).shuffle(row_ids)
    n_val_rows = min(len(row_ids) - 1, max(1, round(len(row_ids) * args.val_fraction)))
    val_rows = set(row_ids[:n_val_rows])
    train = [r for rid in row_ids if rid not in val_rows for r in by_row[rid]]
    val = [r for rid in row_ids if rid in val_rows for r in by_row[rid]]

    os.makedirs(args.out_dir, exist_ok=True)
    n_train = _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    n_val = _write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    print(
        f"traces: {len(conversations)} rendered "
        f"(dropped: {dropped['partial']} partial, {dropped['unrenderable']} unrenderable, "
        f"{n_image_missing} missing-image, {n_overlong} overlong targets)"
    )
    print(f"images: {len(mapping)} extracted -> {os.path.join(args.out_dir, 'images')}")
    print(f"wrote {n_train:6d} records ({len(row_ids) - n_val_rows} rows) -> train.jsonl")
    print(f"wrote {n_val:6d} records ({n_val_rows} rows) -> val.jsonl")


if __name__ == "__main__":
    main()
