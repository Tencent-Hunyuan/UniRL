# MiniMax-H3 model contract

This package implements the FL2VA checkpoint's `t2va` task: text to joint
video and stereo audio. Keyframe-conditioned `fl2va` shares the partition but
is not wired as a UniRL recipe yet; `ref2va` uses a separate partition.

## Checkpoint semantics

- Guidance is distilled into the weights. There is no CFG branch or negative
  prompt, and each denoising step performs one transformer forward.
- The transformer predicts data-ward velocity:
  `x0 = x_t + sigma * v`. UniRL's flow strategy assumes the opposite sign, so
  `predict_noise` negates the H3 velocity once before using the shared SDE,
  log-probability, and replay implementation.
- Video and audio use separate static rectified-flow schedules. The default
  shifts are 12 for video and 3 for audio. The generated Part stores the video
  schedule; the audio schedule is deterministically reconstructed from the same
  step count.
- The checkpoint is mixed precision. Patch projections, timestep layers, and
  output heads remain fp32 while the transformer block stack is bf16.

## Packed joint sequence

H3 denoises text, audio, and video rows in one packed sequence. Video state is
stored in `LatentSegment.latents`; audio state is stored in `aux_latents`.
Per-row modality metadata controls the update masks and timestep assignment.

The implementation keeps the leading batch dimension through patchification
and replay. Audio channel count and audio latent width are distinct concepts:
the decoded signal is stereo, while the packed latent feature width is 32.

## Rollout and replay

The rollout captures only the trajectory positions required by the configured
SDE step indices. Replay supplies each stored video and audio state to the same
joint transformer because either modality's velocity depends on the other.
Per-stream log-probabilities are combined by latent element count, preserving
the scale of the shared video-only objective.

MiniMax-H3 is guidance-distilled and does not tolerate the off-schedule noise
of a high-eta FlowSDE sampler. The recipe uses CPS, which reallocates variance
within the target schedule.

## Text and VAE components

The text conditioner uses Qwen3-VL hidden states from layer 50. It may reside on
CPU independently of the train device; token tensors are created on the
conditioner's own device and the resulting embeddings move to the DiT device.

The video VAE decodes channel-first video latents. The audio VAE emits
length-first stereo samples. Both VAE implementations and the transformer
reference code are vendored from the checkpoint integration recorded in
`vendor/VENDOR_COMMIT.txt`.

## Meta initialization

`finalize_meta_init(..., keep_in_fp32=...)` preserves the checkpoint's mixed
dtypes before `to_empty` materialization. The recipe keeps `root_wrap: false`
so fp32 modules remain outside block-level FSDP mixed-precision groups.
