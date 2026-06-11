#!/usr/bin/env python3
"""Patch a HunyuanImage-3 checkpoint's trust_remote_code files for transformers
5.x compatibility, so the unirl bundle's standalone
``HunyuanImage3Pipeline.generate()`` path runs end-to-end (found via
scripts/hi3_modality_smoke.py, LIN-415).

Handles BOTH checkpoint snapshot styles:
  - base ``HunyuanImage-3``           (modeling file ``hunyuan.py``)
  - ``HunyuanImage-3-Instruct``       (modeling file ``modeling_hunyuan_image_3.py``)

Patches:
  A. KV-cache: transformers 5.x ``StaticLayer.lazy_initialization`` now requires
     ``(key_states, value_states)``; the checkpoint calls it with ``key_states``
     only. (both snapshots)
  B. Siglip2 vision processor returns list-valued ``pixel_values`` in
     transformers 5.x; ``.squeeze(0)`` then fails. Force ``return_tensors="pt"``.
     (anchor differs: base ``vision_encoder_processor`` vs Instruct
     ``vit_info.processor``)
  C. base ``ForCausalMM.forward`` has no ``**kwargs`` and rejects the
     ``rope_image_info`` kwarg unirl's diffusion passes; the forward derives rope
     from ``custom_pos_emb`` so accept+ignore it. (base only — Instruct's forward
     already declares ``rope_image_info``)

Idempotent. Operate on a WRITABLE copy (e.g. /dockerdata/...), not read-only ceph.
After patching, clear the trust_remote_code cache so the patched files reload:
    rm -rf ~/.cache/huggingface/modules/transformers_modules/HunyuanImage*

Usage: python hi3_ckpt_transformers5_compat.py [CKPT_DIR]   # default /dockerdata/HunyuanImage-3
"""
import os
import sys

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/dockerdata/HunyuanImage-3"


def patch(path: str, old: str, new: str, label: str):
    if not os.path.exists(path):
        return None
    s = open(path).read()
    if new in s and old not in s:
        print(f"[compat] {label}: already patched")
        return True
    if s.count(old) == 0:
        print(f"[compat] {label}: anchor not found in {os.path.basename(path)}")
        return False
    open(path, "w").write(s.replace(old, new))
    print(f"[compat] {label}: patched in {os.path.basename(path)}")
    return True


print(f"[compat] patching checkpoint at {CKPT}")
IP = os.path.join(CKPT, "image_processor.py")

# Modeling file differs by snapshot.
modeling = None
for cand in ("hunyuan.py", "modeling_hunyuan_image_3.py"):
    if os.path.exists(os.path.join(CKPT, cand)):
        modeling = os.path.join(CKPT, cand)
        break
if modeling is None:
    print("[compat] ERROR: no modeling file (hunyuan.py / modeling_hunyuan_image_3.py) found")
    sys.exit(3)
print(f"[compat] modeling file: {os.path.basename(modeling)}")

# A — KV-cache lazy_initialization (both snapshots)
patch(
    modeling,
    "self.layers[layer_idx].lazy_initialization(key_states)",
    "self.layers[layer_idx].lazy_initialization(key_states, value_states)",
    "A/kv-cache lazy_initialization",
)

# B — Siglip2 image proc list -> tensor (whichever anchor the snapshot uses)
patch(
    IP,
    "inputs = self.vision_encoder_processor(image)",
    'inputs = self.vision_encoder_processor(image, return_tensors="pt")',
    "B/image-proc return_tensors (base)",
)
patch(
    IP,
    "inputs = self.vit_info.processor(image)",
    'inputs = self.vit_info.processor(image, return_tensors="pt")',
    "B/image-proc return_tensors (instruct)",
)

# C — base forward() accept+ignore rope_image_info (Instruct already declares it)
patch(
    modeling,
    "            first_step: Optional[bool] = None,\n            # for gen image",
    "            first_step: Optional[bool] = None,\n"
    "            rope_image_info: Optional[Any] = None,  # accept+ignore: rope from custom_pos_emb\n"
    "            # for gen image",
    "C/forward accept rope_image_info (base only)",
)

print("[compat] done")
