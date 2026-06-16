"""Offline checkpoint tools: LoRA merge + Hugging Face export.

File-to-file counterparts of the runtime LoRA merging in
``unirl.utils.peft_merge`` (which engines and weight sync use on live
modules). Entry points: ``python -m unirl.tools.export_full`` for a merged
model and ``python -m unirl.tools.export_adapter`` for a PEFT adapter.
"""
