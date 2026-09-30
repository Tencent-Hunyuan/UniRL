<div align="center">

<img src="assets/banner.png" alt="UniRL — A Reinforcement Learning Framework for Unified Multimodal Models" width="98%">

### A Reinforcement Learning Framework for Unified Multimodal Models

**U**(you)·**ni**(need)·**RL** for unified multimodal intelligence

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-unirl--project.github.io-blue)](https://unirl-project.github.io/unirl/)
[![WeChat](https://img.shields.io/badge/WeChat-微信群-07C160?logo=wechat&logoColor=white)](https://unirl-project.github.io/unirl/community/wechat-qr.jpg)

<br>
<a href="https://trendshift.io/repositories/48953" target="_blank" rel="noopener noreferrer">
  <img src="https://trendshift.io/api/badge/trendshift/repositories/48953/daily?language=Python" alt="UniRL ranked #16 Python Repository of the Day on Trendshift" width="250" height="55">
</a>

</div>

## News 🚀

- **[2026-06]** **DRPO** released — *"Rethinking the Divergence Regularization in LLM RL"* ([arXiv](https://arxiv.org/abs/2606.09821)).
- **[2026-06]** **Flow-DPPO** released — *"FlowDPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* ([arXiv](https://arxiv.org/abs/2606.11025)).
- **[2026-06]** **CPPO** released — *"Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning"* ([arXiv](https://arxiv.org/abs/2606.10968)).

## About 💡

UniRL applies one RL post-training loop — generate samples, score them, compute
advantages, update the policy, and sync weights back to rollout workers —
across multimodal model families.

<div align="center">
  <img src="assets/UniRL_arch_new.png" alt="UniRL architecture" width="900">
</div>

UniRL is a layered, composable system. Each **training entrypoint** loads a
**Hydra example config** and creates the matching domain **trainer**. RL trainers
coordinate generation, scoring, and updates across pluggable
**rollout engines**, **algorithms**, **model bundles**, **reward services**, and
the shared **distributed runtime**: Ray `DevicePool`, FSDP, Transfer
Queue (TQ), and LoRA/full-weight sync. SFT consumes supervised manifests, while
the async AR and diffusion entrypoints overlap rollout with training on separate
GPU slabs. See [`examples/README.md`](examples/README.md#domains--entrypoints)
for all entrypoints and [`unirl/README.md`](unirl/README.md) for the runtime loop,
deployment modes, and module map.

## Team-Proposed Algorithms 🌟

> **🌟 These algorithms are proposed by our team — the highlight of UniRL.** Each
> algorithm's folder holds a step-by-step tutorial and a runnable example recipe.
> We highly recommend trying them in our framework!

| Algorithm | Paper | Tutorial | Notes |
|---|---|---|---|
| **Flow-DPPO** | [*"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"*](https://arxiv.org/abs/2606.11025) | [FlowDPPO/](FlowDPPO/) | Diffusion/flow RL with an exact divergence-based trust-region mask. |
| **DRPO** | [*"Rethinking the Divergence Regularization in LLM RL"*](https://arxiv.org/abs/2606.09821) | [DRPO/](DRPO/) | Token-level LLM RL with a smooth advantage-weighted quadratic regularizer. |
| **CPPO** | [*"Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning"*](https://arxiv.org/abs/2606.10968) | [CPPO/](CPPO/) | Token-level LLM RL with a position-weighted, cumulative-prefix-budget Binary-TV mask. |

UniRL also wires in standard reference algorithms — **(LLM's)GRPO**, **DiffusionNFT**,
**DanceGRPO**, and **MixGRPO** — in [`unirl/algorithms/`](unirl/algorithms/README.md).

## Model Support 🎨

Model and algorithm support are **two independent dimensions** that compose within
a domain: any diffusion algorithm (see above) runs on a diffusion
model, AR algorithms on AR models — so UniRL covers many more model × algorithm
combinations than the shipped example recipes alone. The table below is the model
dimension: each row links one runnable recipe. Full matrix — package, every recipe,
rollout engine, and restriction per model — in
[`unirl/models/README.md`](unirl/models/README.md#support-matrix).

<!-- MiniMax-H3 is trainside-only until the vLLM-Omni rollout backend (#378 / #420)
     lands; update its row here and in unirl/models/README.md when it merges. -->

<div align="center">

| Model | Category | Modality | Recipe | Status |
|---|---|---|---|---|
| Stable Diffusion 3.5 | Image diffusion | Text → Image | [`sd3_trainside`](examples/diffusion/sd3/sd3_trainside.yaml) | ✅ |
| Qwen-Image | Image diffusion | Text → Image | [`qwen_image_trainside`](examples/diffusion/qwen_image/qwen_image_trainside.yaml) | ✅ |
| Qwen-Image-Edit-2511 | Image diffusion | Text + Image → Image | [`qwen_image_edit_plus_nft`](examples/diffusion/qwen_image_edit_plus/qwen_image_edit_plus_nft.yaml) | ✅ |
| FLUX.2-Klein (4B / 9B) | Image diffusion | Text → Image / Text + Image → Image | [`flux2_klein_trainside`](examples/diffusion/flux2_klein/flux2_klein_trainside.yaml) | ✅ |
| Z-Image | Image diffusion | Text → Image | [`z_image_trainside`](examples/diffusion/z_image/z_image_trainside.yaml) | ✅ |
| Boogu-Image-0.1 | Image diffusion | Text → Image | [`boogu_image_trainside`](examples/diffusion/boogu_image/boogu_image_trainside.yaml) | ✅ trainside only |
| WAN 2.1 | Video diffusion | Text / Image → Video | [`wan21_t2v`](examples/diffusion/wan21/wan21_t2v.yaml) | ✅ |
| WAN 2.2 (A14B) | Video diffusion | Text / Image → Video | [`wan22_t2v_14b`](examples/diffusion/wan22/wan22_t2v_14b.yaml) | ✅ |
| WAN 2.2 V2V | Video diffusion | Video → Video | [`wan22_v2v_14b`](examples/diffusion/wan22_v2v/wan22_v2v_14b.yaml) | ✅ trainside only |
| HunyuanVideo 1.0 | Video diffusion | Text → Video | [`hunyuan_video10_t2v_trainside`](examples/diffusion/hunyuan_video10/hunyuan_video10_t2v_trainside.yaml) | ✅ |
| HunyuanVideo 1.5 | Video diffusion | Text → Video | [`hunyuan_video15_t2v_dancegrpo_trainside`](examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_dancegrpo_trainside.yaml) | ✅ |
| LTX-2 | Video diffusion | Text → Video | [`ltx2_t2v_trainside`](examples/diffusion/ltx2/ltx2_t2v_trainside.yaml) | ✅ |
| LTX-2.3 | Video diffusion | Text → Audio + Video | [`ltx2_3_t2av_trainside`](examples/diffusion/ltx2/ltx2_3_t2av_trainside.yaml) | ✅ trainside only |
| MiniMax-H3 | Video diffusion | Text → Video + Audio | [`minimax_h3_t2va_trainside`](examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml) | ✅ trainside only |
| HunyuanImage 3.0 | Unified AR + diffusion | Text / Text + Image → Image | [`hi3_trainside_t2i`](examples/unified_model/hi3_trainside_t2i.yaml) | ✅ |
| BAGEL-7B-MoT | Unified AR + diffusion | Text / Text + Image → Image; Text + Image → Text | [`bagel_trainside_lora`](examples/diffusion/bagel/bagel_trainside_lora.yaml) | ✅ |
| SenseNova-U1.5 | Unified MoT pixel flow | Text → Image | [`sensenova_u1_5_trainside`](examples/diffusion/sensenova_u1_5/sensenova_u1_5_trainside.yaml) | ✅ trainside only |
| Janus-Pro | Unified AR | Text → Image; Text + Image → Text | [`janus_pro_grpo_t2i_lora`](examples/ar/janus_pro_grpo_t2i_lora.yaml) | ✅ trainside only |
| Qwen3 | LLM AR | Text → Text | [`qwen3_grpo_4b_base_dapo_sglang`](examples/ar/qwen3_grpo_4b_base_dapo_sglang.yaml) | ✅ |
| Qwen3-MoE (VeOmni EP) | LLM AR | Text → Text | [`qwen3_moe_grpo_30b_a3b_veomni_ep_sglang`](examples/ar/qwen3_moe_grpo_30b_a3b_veomni_ep_sglang.yaml) | 🧩 bundle-only |
| Qwen3.5 (9B / 35B-A3B) | VLM AR | Text / Text + Image → Text | [`qwen3_5_grpo_9b_base_dapo_sglang`](examples/ar/qwen3_5_grpo_9b_base_dapo_sglang.yaml) | ✅ sglang only |
| Qwen2.5-VL | VLM AR | Text + Image → Text | [`qwen_vl_grpo_geo3k_mc_4x8`](examples/ar/qwen_vl_grpo_geo3k_mc_4x8.yaml) | ✅ |
| Qwen3-Omni Thinker | Omni-modality AR | Text / Image / Audio / Video → Text | [`qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4`](examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4.yaml) | ✅ vllm_omni only |
| Cosmos3-Nano | World model | Video (+ action) prediction | [`sft/cosmos3_droid100_videopred`](examples/sft/cosmos3_droid100_videopred.yaml) | 🧪 SFT-only |
| Prompt-Enhancer | LLM + diffusion (composed) | Text → Text → Image | [`pe_trainside_pickscore`](examples/pe/pe_trainside_pickscore.yaml) | 🔗 composed |

</div>

✅ runnable end-to-end with the linked recipe (a qualifier names the only rollout
engine that has a recipe) · 🧩 bundle-only (no pipeline of its own; runs under
another package's pipeline) · 🧪 SFT-only (no rollout path) · 🔗 composed from other
rows. SFT, async, agentic, and scorer-service requirements per model are in the
[full matrix](unirl/models/README.md#support-matrix).

## Training Modes 🧩

Each entrypoint has one built-in default, used when `--config-name` is omitted.

| Training path | Trains | Entrypoint | Built-in default recipe |
|---|---|---|---|
| Diffusion RL | Image / video diffusion models | `train_diffusion` | [`diffusion/sd3/sd3_trainside`](examples/diffusion/sd3/sd3_trainside.yaml) |
| AR RL | Vision-language (VLM) + text-only (LLM) models | `train_ar` | [`ar/qwen_vl_grpo_geo3k_mc_4x8`](examples/ar/qwen_vl_grpo_geo3k_mc_4x8.yaml) |
| SFT | Supervised text, multimodal, and diffusion models | `train_sft` | [`sft/qwen3_sft`](examples/sft/qwen3_sft.yaml) |
| Prompt enhancement | AR rewriter + diffusion reward | `train_pe` | [`pe/pe_trainside_pickscore`](examples/pe/pe_trainside_pickscore.yaml) |
| Unified RL | Unified AR + diffusion models | `train_unified_model` | [`unified_model/hi3_vllmomni`](examples/unified_model/hi3_vllmomni.yaml) |
| Agentic RL | Service-scored multi-turn tool use | `train_agentic` | [`deep_research/deep_research_search_judge`](examples/deep_research/deep_research_search_judge.yaml) |
| Async AR RL | AR models with separate train / rollout workers | `train_async_ar` | [`ar/qwen3_grpo_4b_base_dapo_sglang_async`](examples/ar/qwen3_grpo_4b_base_dapo_sglang_async.yaml) |
| Async diffusion RL | Diffusion models with separate train / rollout workers | `train_async_diffusion` | [`diffusion/bagel/bagel_vllmomni_async`](examples/diffusion/bagel/bagel_vllmomni_async.yaml) |

See [`examples/README.md`](examples/README.md) for the full launch guide, naming
schema, and how to add a recipe.

## Agentic Workflows 🤖

`train_agentic` extends the AR path with multi-turn tool use. Each turn is a
`Sample` in a lineage; terminal answers are scored by a reward service, and
training waits at a colocated rollout barrier.

`AgenticTrainer` synchronizes current training weights before every rollout,
dispatches sibling trajectories concurrently, and waits for complete GRPO groups
before scoring and training. Each successful trajectory receives one group-normalized
advantage, which is applied to every generated assistant turn. Failed
trajectories are excluded from the update.

See the [agent environment guide](unirl/rollout/env/README.md) for the
environment, tool, and trajectory contracts.

## Getting Started ⚡

Install dependencies first — see [INSTALL.md](INSTALL.md).

```bash
# compose-check, then launch a single-node example
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
```

Full [launch guide](examples/README.md#running-a-recipe) — multi-node, every entrypoint, mooncake.

## Roadmap 🗺️

We are actively expanding model and algorithm coverage. Near-term directions:

- Broaden algorithm coverage for the newer model families — FLUX.2-Klein,
  HunyuanVideo 1.0 / 1.5, and Bagel.
- Extend the team-proposed algorithms (Flow-DPPO, DRPO) to more model families.
- Broaden reward backends and rollout-engine coverage across domains.

Want a model or algorithm prioritized? Open a
[feature request](https://github.com/Tencent-Hunyuan/UniRL/issues/new?template=feature-request.yml)
to discuss.

## Contributing 🤝

Contributions and questions are welcome. Before opening a pull request, read the
repository conventions in [`AGENTS.md`](AGENTS.md), run the
[pre-PR checks](examples/README.md#adding-or-editing-a-recipe) for the files you
touched, and fill in the [pull request template](.github/pull_request_template.md).
Use the issue forms for a
[bug report](https://github.com/Tencent-Hunyuan/UniRL/issues/new?template=bug-report.yml)
or
[feature request](https://github.com/Tencent-Hunyuan/UniRL/issues/new?template=feature-request.yml).
WeChat is fine for chat; bugs still belong on GitHub so they stay searchable.

## Acknowledgement 🙏

UniRL builds on ideas and infrastructure from the open-source RL and inference
ecosystem. We especially thank
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[slime](https://github.com/THUDM/slime), and
[verl](https://github.com/volcengine/verl).

## Citation 📚

If you find UniRL helpful, please cite:

```bibtex
@misc{unirl_github,
  title        = {{UniRL: A Reinforcement Learning Framework for Unified Multimodal Models}},
  author       = {Haonan Wang and Linyu Wu and Qian Qiu and Lewei Jin and Bowen Ping and Jianghai Chen and Yiheng Du and Guangxin He and Yu Shi and Yongguang Lin and Zhuoxin Zhou and Zhanchao Zhou and Keming Wu and Rizhen Hu and Xuefei Ning and Lvfang Tao and Feiyu Hu and Xiangyan Liu and Siqi Kou and Jiarui Yao and Xiangxin Zhou and Liefeng Bo and Wenxi Zhu and Tianyu Pang},
  year         = {2026},
  howpublished = {\url{https://github.com/Tencent-Hunyuan/UniRL}},
  urldate      = {2026-06-05}
}
```

If you use DRPO, please also cite:

```bibtex
@article{yao2026rethinking,
  title={Rethinking the Divergence Regularization in LLM RL},
  author={Yao, Jiarui and Zhou, Xiangxin and Qi, Penghui and Lee, Wee Sun and Bo, Liefeng and Pang, Tianyu},
  journal={arXiv preprint arXiv:2606.09821},
  year={2026}
}
```

If you use Flow-DPPO, please also cite:

```bibtex
@article{ping2026flow,
  title={Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models},
  author={Ping, Bowen and Zhou, Xiangxin and Qi, Penghui and Luo, Minnan and Bo, Liefeng and Pang, Tianyu},
  journal={arXiv preprint arXiv:2606.11025},
  year={2026}
}
```
