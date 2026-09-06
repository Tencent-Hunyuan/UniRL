# OCR text-rendering prompts

These prompts come from the OCR split imported from
[X-GenGroup/Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) under Apache-2.0.
Each prompt contains one quoted target string consumed by
`unirl.reward.local.ocr.OCRRewardScorer`.

`train.txt` and `test.txt` are the full prompt sets. `short_text_train.txt` is an
order-preserving subset of `train.txt` for low-resolution reward-curve validation: it
retains targets containing one or two ASCII alphabetic words and between three and
eight non-space characters.

`examples/diffusion/sensenova_u1_5/sensenova_u1_5_ocr_trainside_lora_4x8.yaml` trains on
the short split and evaluates on the full `test.txt`, so the reported curve is measured
against unrestricted target lengths rather than the training subset's own distribution.
