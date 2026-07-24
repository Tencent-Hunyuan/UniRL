"""Per-model EP wiring for the VeOmni backend.

Each module here adapts one model family's MoE to VeOmni expert parallelism
(fused-expert module + checkpoint converter + parallel plan + the meta-swap),
and feeds the generic loader in :mod:`unirl.train.backend.veomni.ep` (which
shards the fused expert weights model-agnostically). e.g. :mod:`.hi3` for
HunyuanImage 3.0.
"""
