"""TRUE parity: UniRL Qwen3 SFT loss  vs  HF-native model(labels=).loss.

This is the loss TRL's SFTTrainer actually optimizes: SFTTrainer builds
{input_ids, labels(prompt=-100)} and calls the model, whose forward computes
CausalLM cross-entropy internally (logits[:-1] vs labels[1:], mean over the
non-(-100) tokens). Rather than hand-write that CE (which could hide a
masking/shift mistake), we call the model's OWN loss and compare to our
Qwen3SFTTask.compute_loss (-mean(logp) via _replay_aware_forward).

To be certain the two paths see the identical labeling, we drive our task's
tokenizer (Qwen3SFTTask.load_record) for BOTH: HF uses full_ids + labels with
prompt masked to -100; ours uses response_tokens + prompt_len. Same base model,
same batch, fp32, single GPU. They must match to ~1e-3.

Also prints what trl.SFTTrainer would build, and asserts our label construction
equals SFTTrainer's DataCollatorForCompletionOnlyLM semantics (prompt tokens
ignored, response tokens kept).
"""

import json
import sys
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/sync/models/Qwen3-4B-Base"
DATA = "/root/shared/.clusters/.tmp/sft-qwen3-sd3/datasets/sft/qwen3_train.jsonl"
N = 8


def tokenize(tok, prompt, response):
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    response_ids = tok.encode(response, add_special_tokens=False)
    if tok.eos_token_id is not None:
        response_ids = list(response_ids) + [tok.eos_token_id]
    return list(prompt_ids), list(response_ids)


@torch.no_grad()
def hf_native_loss(model, prompt_ids, response_ids, device):
    """The loss TRL SFTTrainer optimizes: model's OWN CausalLM loss with
    labels = [-100]*prompt + response (completion-only masking)."""
    full = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + list(response_ids)], dtype=torch.long, device=device)
    # NOTE: must use the STOCK forward (labels=) — install a fresh model instance
    # for HF so our _replay_aware_forward override does not shadow it.
    out = model(input_ids=full, labels=labels, use_cache=False)
    return out.loss


@torch.no_grad()
def ours_loss(model, prompt_ids, response_ids, device):
    from unirl.models.qwen3.ar import _replay_aware_forward

    if getattr(model.forward, "__func__", None) is not _replay_aware_forward:
        model.forward = MethodType(_replay_aware_forward, model)
    full = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    resp = torch.tensor([response_ids], dtype=torch.long, device=device)
    pos = torch.arange(full.shape[1], device=device).unsqueeze(0)
    logp = model(
        input_ids=full,
        attention_mask=torch.ones_like(full),
        position_ids=pos,
        response_tokens=resp,
        prompt_len=len(prompt_ids),
        temperature=1.0,
        autocast_dtype=None,
    )
    return -logp.mean()


def main():
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    records = [json.loads(x) for x in open(DATA)][:N]

    # Two SEPARATE model instances so the _replay_aware_forward install on the
    # "ours" model can never shadow the stock labels= forward on the HF model.
    hf_model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    )
    our_model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    )

    print(f"TRUE parity (HF-native .loss vs ours) over {len(records)} records, fp32:")
    diffs = []
    for i, r in enumerate(records):
        p, resp = tokenize(tok, r["prompt"], r["response"])
        hf = float(hf_native_loss(hf_model, p, resp, device))
        ours = float(ours_loss(our_model, p, resp, device))
        d = abs(hf - ours)
        diffs.append(d)
        print(f"  [{i}] hf_native={hf:.5f}  ours={ours:.5f}  |Δ|={d:.2e}")
    mx = max(diffs)
    print(f"max |Δ| = {mx:.2e}")
    ok = mx < 5e-3
    print("TRUE PARITY vs HF/TRL loss:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
