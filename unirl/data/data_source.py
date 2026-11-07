"""Data source implementations for GRPO training."""

import logging
import os
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader

from unirl.types.media import MediaRef, MediaRefs
from unirl.types.primitives import Image, Images, Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.utils.video import load_video

from .datasets import TextPromptDataset

logger = logging.getLogger(__name__)


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {
    ("image", "condition"),
    ("image", "prompt"),
    ("video", "condition"),
    ("audio", "prompt"),
    ("video", "prompt"),
}


class MultimodalRLDataSource:
    """Multimodal runtime data source for RL training."""

    def __init__(self, args):
        """Initialize data source from arguments."""
        self.args = args
        self.data_path = args.run.data_path
        self.eval_data_path = args.run.eval_data_path
        self.seed = args.run.seed
        self.shuffle = bool(getattr(args.run, "shuffle", True))
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        self.eval_dataset: Optional[TextPromptDataset] = None
        self._shuffle_generator = torch.Generator()
        if self.seed is None:
            _shuffle_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
        else:
            _shuffle_seed = int(self.seed)
        self._shuffle_generator.manual_seed(_shuffle_seed)

        if not self.data_path:
            raise ValueError("MultimodalRLDataSource requires args.run.data_path.")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data path not found: {self.data_path}")

        self.train_dataset = TextPromptDataset(file_path=self.data_path)
        logger.info(
            "Loaded multimodal training dataset from %s (%d samples)",
            self.data_path,
            len(self.train_dataset),
        )
        if len(self.train_dataset) < self.prompts_per_rollout:
            raise ValueError(
                "Training dataset is smaller than prompts_per_rollout, which would produce an "
                f"empty DataLoader with drop_last=True (num_prompts={len(self.train_dataset)}, "
                f"prompts_per_rollout={self.prompts_per_rollout})."
            )
        self._dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.prompts_per_rollout,
            shuffle=self.shuffle,
            generator=self._shuffle_generator if self.shuffle else None,
            num_workers=0,
            collate_fn=self._prompt_examples_to_batch,
            drop_last=True,
        )
        self._iter: Iterator = iter(self._dataloader)

    @property
    def num_prompts(self) -> int:
        """Total number of prompts in the training dataset."""
        return len(self.train_dataset)

    def _prompt_examples_to_batch(self, prompt_examples: List[Dict[str, Any]]) -> Sample:
        """Convert normalized prompt examples into an input-only ``Sample``."""
        prompts = [item["prompt"] for item in prompt_examples]
        sample_ids = [f"prompt:{item['prompt_id']}:sample:0" for item in prompt_examples]

        primitives: Dict[str, Any] = {}
        media_refs: List[List[MediaRef]] = [item.get("media_refs", []) for item in prompt_examples]
        if any(media_refs):
            video_roles = {ref.role for refs in media_refs for ref in refs if ref.modality == "video"}
            if "condition" in video_roles and "prompt" in video_roles:
                raise ValueError("A batch cannot mix (video, condition) and (video, prompt) MediaRefs.")
            bad = [
                (i, r)
                for i, refs in enumerate(media_refs)
                for r in refs
                if (r.modality, r.role) not in _SUPPORTED_MEDIA_REF_ROLES
            ]
            if bad:
                raise NotImplementedError(
                    f"MultimodalRLDataSource._prompt_examples_to_batch: media_refs include {len(bad)} unsupported "
                    f"(modality, role) entries; supported pairs are {sorted(_SUPPORTED_MEDIA_REF_ROLES)}. "
                    f"First bad entry: prompt={bad[0][0]}, ref={bad[0][1]!r}."
                )

            for modality in ("image", "video"):
                cond_uris = [
                    next((r.uri for r in refs if r.modality == modality and r.role == "condition"), None)
                    for refs in media_refs
                ]
                if not any(cond_uris):
                    continue
                missing = [i for i, uri in enumerate(cond_uris) if uri is None]
                if missing:
                    raise ValueError(
                        f"Condition-{modality} batch is incomplete: {len(missing)}/{len(cond_uris)} "
                        f"prompts are missing a condition {modality} (e.g. prompt index {missing[0]})."
                    )
                if modality == "image":
                    import PIL.Image
                    import torchvision.transforms.functional as TF

                    primitives["image"] = Images.from_list(
                        [Image(pixels=TF.to_tensor(PIL.Image.open(uri).convert("RGB"))) for uri in cond_uris]
                    )
                else:
                    primitives["video"] = Videos.from_list([Video(frames=load_video(uri)) for uri in cond_uris])

            prompt_rows = [[ref for ref in refs if ref.role == "prompt"] for refs in media_refs]
            if any(prompt_rows):
                primitives["media"] = MediaRefs.from_rows(prompt_rows)

        root = Part.input(
            sample_ids,
            primitives={"text": Texts(texts=prompts)},
            metadata=[item.get("metadata") for item in prompt_examples],
        )
        parts = [root]
        parent = root
        for key in ("image", "video", "media"):
            primitive = primitives.get(key)
            if primitive is not None:
                parent = parent.input_child({key: primitive})
                parts.append(parent)
        return Sample.request(*parts)

    def get_samples(self, batch_size: int) -> Sample:
        """Get the next batch as an input-only request ``Sample``."""
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._dataloader)
            return next(self._iter)

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[Sample]:
        """Yield the evaluation prompt source in deterministic batches."""
        batch_size = int(batch_size)
        eval_num_prompts = int(eval_num_prompts)
        if batch_size <= 0 or eval_num_prompts == 0:
            return
        if self.eval_dataset is None:
            if self.eval_data_path:
                if not os.path.exists(self.eval_data_path):
                    raise FileNotFoundError(f"Evaluation data path not found: {self.eval_data_path}")
                eval_path = self.eval_data_path
                source_label = "eval_data_path"
            else:
                eval_path = self.data_path
                source_label = "data_path"
            self.eval_dataset = TextPromptDataset(file_path=eval_path)
            logger.info(
                "Loaded evaluation prompt source from %s=%s (%d examples, deterministic order)",
                source_label,
                eval_path,
                len(self.eval_dataset),
            )

        total = len(self.eval_dataset)
        limit = total if eval_num_prompts < 0 else min(eval_num_prompts, total)
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            prompt_examples = [self.eval_dataset.get_prompt_example(idx) for idx in range(start, end)]
            yield self._prompt_examples_to_batch(prompt_examples)

    def get_eval_samples(self, batch_size: int) -> Sample:
        """Return the first eval batch (BC shim over :meth:`iter_eval_batches`)."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._prompt_examples_to_batch([])
        return next(
            self.iter_eval_batches(batch_size),
            self._prompt_examples_to_batch([]),
        )


class MultiDomainRLDataSource:
    """Round-robin multi-domain prompt source — one domain per rollout batch."""

    def __init__(self, args):
        run_cfg = args.run
        entries = list(getattr(run_cfg, "domains", None) or ())
        if not entries:
            raise ValueError("MultiDomainRLDataSource requires args.run.domains (a non-empty list).")

        self.domains: List[Dict[str, Optional[str]]] = []
        for entry in entries:
            name, data_path, eval_data_path = entry.get("name"), entry.get("data_path"), entry.get("eval_data_path")
            if not name or not data_path:
                raise ValueError(f"Each domain entry needs 'name' and 'data_path'; got {entry!r}.")
            if not os.path.exists(str(data_path)):
                raise FileNotFoundError(f"Domain {name!r} data_path not found: {data_path}")
            if eval_data_path and not os.path.exists(str(eval_data_path)):
                raise FileNotFoundError(f"Domain {name!r} eval_data_path not found: {eval_data_path}")
            self.domains.append(
                {
                    "name": str(name),
                    "data_path": str(data_path),
                    "eval_data_path": str(eval_data_path) if eval_data_path else None,
                }
            )
        names = [d["name"] for d in self.domains]
        if len(set(names)) != len(names):
            raise ValueError(f"Domain names must be unique, got {names}.")

        self.seed = getattr(run_cfg, "seed", None)
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        self._datasets: List[TextPromptDataset] = []
        self._dataloaders: List[DataLoader] = []
        self._iters: List[Iterator] = []
        self._eval_datasets: Optional[List[TextPromptDataset]] = None
        self._iter_counter = 0

        for i, domain in enumerate(self.domains):
            ds = TextPromptDataset(file_path=domain["data_path"])
            if len(ds) < self.prompts_per_rollout:
                raise ValueError(
                    f"Domain {domain['name']!r} dataset is smaller than prompts_per_rollout, which "
                    f"would produce an empty DataLoader with drop_last=True "
                    f"(num_prompts={len(ds)}, prompts_per_rollout={self.prompts_per_rollout})."
                )
            generator = torch.Generator()
            if self.seed is None:
                generator.manual_seed(int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF)
            else:
                generator.manual_seed(int(self.seed) + i)
            loader = DataLoader(
                ds,
                batch_size=self.prompts_per_rollout,
                shuffle=True,
                generator=generator,
                num_workers=0,
                collate_fn=partial(self._domain_examples_to_batch, domain["name"]),
                drop_last=True,
            )
            self._datasets.append(ds)
            self._dataloaders.append(loader)
            self._iters.append(iter(loader))
            logger.info(
                "MultiDomainRLDataSource: domain %r — %d prompts from %s",
                domain["name"],
                len(ds),
                domain["data_path"],
            )

    def _domain_examples_to_batch(self, domain: str, examples: List[Dict[str, Any]]) -> Sample:
        """Build a Sample from one domain's prompt examples, domain-stamped."""
        if any(item.get("media_refs") for item in examples):
            raise ValueError(
                f"MultiDomainRLDataSource is text-prompt-only; domain {domain!r} has media_refs. "
                "Extend it alongside MultimodalRLDataSource._prompt_examples_to_batch when a recipe needs media."
            )
        prompts = [item["prompt"] for item in examples]
        sample_ids = [f"prompt:{domain}:{item['prompt_id']}:sample:0" for item in examples]
        metadata_list = [{**item.get("metadata", {}), "domain": domain} for item in examples]
        return Sample.request(Part.input(sample_ids, primitives={"text": Texts(texts=prompts)}, metadata=metadata_list))

    def get_samples(self, batch_size: int) -> Sample:
        """Next single-domain batch; domains cycle in declaration order."""
        idx = self._iter_counter % len(self.domains)
        self._iter_counter += 1
        try:
            return next(self._iters[idx])
        except StopIteration:
            self._iters[idx] = iter(self._dataloaders[idx])
            return next(self._iters[idx])

    @property
    def num_prompts(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    def get_eval_samples(self, batch_size: int) -> Sample:
        """One deterministic eval Sample of up to ``batch_size`` prompts."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._domain_examples_to_batch(self.domains[0]["name"], [])
        if self._eval_datasets is None:
            self._eval_datasets = [
                TextPromptDataset(file_path=domain["eval_data_path"] or domain["data_path"]) for domain in self.domains
            ]
        per_domain, remainder = divmod(batch_size, len(self.domains))
        batches: List[Sample] = []
        for i, (domain, ds) in enumerate(zip(self.domains, self._eval_datasets)):
            limit = min(per_domain + int(i < remainder), len(ds))
            if limit <= 0:
                continue
            examples = [ds.get_prompt_example(idx) for idx in range(limit)]
            batches.append(self._domain_examples_to_batch(domain["name"], examples))
        if not batches:
            return self._domain_examples_to_batch(self.domains[0]["name"], [])
        if len(batches) == 1:
            return batches[0]
        return Sample.concat(batches)
