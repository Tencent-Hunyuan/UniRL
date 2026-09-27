# OCR text-rendering prompts

These prompts come from the OCR split imported from
[X-GenGroup/Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) under Apache-2.0.
Each prompt contains one quoted target string consumed by
`unirl.reward.local.ocr.OCRRewardScorer`.

`train.txt` and `test.txt` are the full prompt sets, used as-is by
`examples/diffusion/sensenova_u1_5/sensenova_u1_5_ocr_trainside_lora_4x8.yaml`. Targets
there run to a median of 15 non-space characters across roughly three words, so the
reported reward curve reflects unrestricted target lengths.
