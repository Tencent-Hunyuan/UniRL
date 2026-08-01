"""Per-model EP wiring for the VeOmni backend.

Each module owns one model family's fused-expert naming and tensor layout:
:mod:`.hi3` includes the HI3 module swap/parallel plan, while
:mod:`.qwen3_moe` is the shared HF ↔ fused converter used by Qwen3 loading and
rollout sync. Transport and DTensor placement remain model-agnostic in the
parent package.
"""
