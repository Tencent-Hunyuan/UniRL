# HunyuanVideo 1.0

> **Where it fits:** text-to-video rollout and training stages. In: LLaMA and
> pooled CLIP conditions. Out: video latent trajectories and decoded frames.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **Embedded guidance uses checkpoint units, not external CFG.**
  `HunyuanVideo10DiffusionStep.predict_noise` multiplies
  `sampling.guidance_scale` by 1000 before the transformer's guidance embedding,
  matching Diffusers `HunyuanVideoPipeline`. The same path serves rollout,
  replay, and ART anchors. A recipe value of 1 supplies 1000 to the transformer.
- **SGLang 0.5.19 does not scale Hunyuan embedded guidance.**
  `HunyuanVideo10Adapter.boot_kwargs` converts the structured sampling scale
  into native `embedded_cfg_scale` units. An explicit
  `rollout.config.engine_kwargs.embedded_cfg_scale` remains a native-unit
  override and is not multiplied again. Remove this conversion when upgrading
  to a SGLang version that performs it internally.
