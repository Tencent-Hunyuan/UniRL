<div align="center">

<img src="assets/banner.png" alt="UniRL — A Reinforcement Learning Framework for Unified Multimodal Models" width="98%">

### A Reinforcement Learning Framework for Unified Multimodal Models

**U**(you)·**ni**(need)·**RL** for unified multimodal intelligence

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-unirl--project.github.io-blue)](https://unirl-project.github.io/unirl/)
[![WeChat](https://img.shields.io/badge/WeChat-微信群-07C160?logo=wechat&logoColor=white)](https://unirl-project.github.io/unirl/community/wechat-qr.jpg)

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

UniRL is a layered, composable system. The library (`unirl/`) is pure components; the
runnable programs live in [`recipes/`](recipes/README.md). Each **recipe**
(`recipes/diffusion`, `recipes/ar`, `recipes/pe`, `recipes/unified_model`, …) is a
self-contained package — a domain **trainer** (`DiffusionTrainer`, `ARTrainer`,
`PETrainer`, `UnifiedModelTrainer`) merged with its `@hydra.main` entrypoint, plus its
**configs** under `configs/`. Launched with `python -m recipes.<task>`, a recipe loads a
config covering model, algorithm, rollout, reward, placement, and sync; the trainer
coordinates the RL loop across pluggable **rollout engines**, **algorithms**, **model
bundles**, **reward services**, and the shared **distributed runtime**: Ray `DevicePool`,
FSDP, Transfer Queue (TQ), and LoRA/full-weight sync. See
[`unirl/README.md`](unirl/README.md) for the runtime loop and module map, and
[`recipes/README.md`](recipes/README.md) for the launch guide.

## Team-Proposed Algorithms 🌟

> **🌟 These algorithms are proposed by our team — the highlight of UniRL.** Each
> algorithm's folder holds a step-by-step tutorial and a runnable example config.
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
combinations than the shipped example configs alone. The table below is the model
dimension; all listed models are supported (✅).

<div align="center">

| Model | Category | Modality | Status |
|---|---|---|---|
| Stable Diffusion 3 / 3.5 | Image diffusion | Text → Image | ✅ |
| Qwen-Image | Image diffusion | Text → Image | ✅ |
| FLUX.2-Klein | Image diffusion | Text → Image / Text + Image → Image | ✅ |
| Z-Image | Image diffusion | Text → Image | ✅ |
| WAN 2.1 | Video diffusion | Text / Image → Video | ✅ |
| WAN 2.2 | Video diffusion | Text / Image → Video | ✅ |
| HunyuanVideo 1.0 / 1.5 | Video diffusion | Text → Video | ✅ |
| LTX-Video-2 | Video diffusion | Text → Video | ✅ |
| LTX-Video-2.3 | Video diffusion | Text → Audio + Video | ✅ |
| Qwen-VL | Vision-language AR | Text + Image → Text | ✅ |
| Qwen3 | LLM AR | Text → Text | ✅ |
| Prompt-Enhancer | LLM + diffusion | Text → Text → Image | ✅ |
| HunyuanImage3 | Unified AR + diffusion | Text → Image | ✅ |
| Bagel | Unified AR + diffusion | Text / Text + Image → Image | ✅ |

</div>

Each model maps to a recipe (`recipes.diffusion`, `recipes.ar`, `recipes.pe`,
`recipes.unified_model`); see **Getting Started** below to run any of them.

## Training Modes 🧩

UniRL unifies four training modes, one **recipe** each. A recipe is launched with
`python -m recipes.<task>` and picks a self-contained config from its own `configs/`
with `--config-name=<config>`:

| Recipe | Trains | Launch | Example config |
|---|---|---|---|
| `diffusion` | Image / video diffusion models | `python -m recipes.diffusion` | `sd3/sd3_sglang_rollout_colocate` |
| `ar` | Autoregressive models — vision-language (VLM) + text-only (LLM) | `python -m recipes.ar` | `qwen_vl_grpo_geo3k_mc_4x8`, `qwen3_drpo_4b_base_dapo_sglang` |
| `pe` | Prompt-enhancer (AR rewriter + diffusion reward) | `python -m recipes.pe` | `pe_sglang_full_pickscore` |
| `unified_model` | Unified AR + diffusion models | `python -m recipes.unified_model` | `hi3_vllmomni` |

`recipes.refl` (differentiable-reward backprop) and `recipes.async_ar` (disaggregated
async AR) round out the set. See [`recipes/README.md`](recipes/README.md) for the full
launch guide, config-name schema, and how to add a recipe.

## Getting Started ⚡

Install dependencies first — see [INSTALL.md](INSTALL.md).

```bash
# compose-check, then launch a single-node example
python -m recipes.diffusion --config-name=sd3/sd3_trainside --cfg job --resolve
bash scripts/run_experiment_single_node.sh sd3/sd3_trainside
```

Full [launch guide](recipes/README.md#running-a-recipe) — multi-node, every recipe, mooncake.

## Roadmap 🗺️

We are actively expanding model and algorithm coverage. Near-term directions:

- Broaden algorithm coverage for the newer model families — FLUX.2-Klein,
  HunyuanVideo 1.0 / 1.5, and Bagel.
- Extend the team-proposed algorithms (Flow-DPPO, DRPO) to more model families.
- Broaden reward backends and rollout-engine coverage across domains.

Want a model or algorithm prioritized? [Open an issue](https://github.com/Tencent-Hunyuan/UniRL/issues) to discuss.

## Contributing 🤝

Contributions and questions are welcome. Before opening a pull request, read the
repository conventions in [`AGENTS.md`](AGENTS.md), run the
[pre-PR checks](recipes/README.md#adding-or-editing-a-config) for the files you
touched, and fill in the [pull request template](.github/pull_request_template.md).
For questions, bug reports, and feature requests,
[open an issue](https://github.com/Tencent-Hunyuan/UniRL/issues).

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
