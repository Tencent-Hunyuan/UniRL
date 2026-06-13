"""Offline checkpoint tools: LoRA merge + Hugging Face export.

File-to-file counterparts of the runtime LoRA merging in
``unirl.utils.peft_merge`` (which engines and weight sync use on live
modules). Entry point: ``python -m unirl.tools.export_hf``.
"""
