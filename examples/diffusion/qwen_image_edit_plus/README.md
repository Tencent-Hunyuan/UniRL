# Qwen-Image-Edit-Plus (Qwen-Image-Edit-2511) recipes

Image-editing RL recipes (DiffusionNFT / FlowGRPO / FlowDPPO) on the
`sglang_diffusion` colocate backend. The model consumes a source image plus an
edit instruction; the source image is VAE-encoded and token-concatenated to the
noise latent inside `predict_noise` (transformer `in_channels=64`).

## Mixed-aspect batches

Upstream sglang's Edit-Plus preprocessing resizes each source image to
~1024x1024 **area while preserving its aspect ratio**, so images with different
aspect ratios produce different latent grid shapes (`H_img x W_img`). UniRL
keeps source images in packed `Images`, captures the resulting condition latents
as a per-sample `QwenImageEditPlusLatentCondition`, and groups equal latent shapes
inside `predict_noise`. The grouped predictions are restored to the original
sample order with a differentiable gather, so rollout and replay support mixed
portrait/landscape/square batches without padding or warping.

Aspect-ratio bucketing in the dataset is optional and can still improve
throughput by reducing the number of transformer microbatches.

## Multiple references

Qwen-Image-Edit-2511 accepts one ordered `ImageSet` per prompt. Text encoding
adds `Picture 1` … `Picture N` placeholders, source VAE latents remain separate
per row until the diffusion step concatenates their tokens, and SGLang rollout
capture reconstructs the same ordered latent rows for replay. Ragged reference
counts and mixed source aspect ratios are grouped by the complete latent-shape
tuple; empty rows fail explicitly.

Set `max_sequence_length` high enough for every visual placeholder. The
multi-reference path raises instead of truncating an overlong row. Legacy
`Images` and singleton `ImageSets` retain the N=1 path.
