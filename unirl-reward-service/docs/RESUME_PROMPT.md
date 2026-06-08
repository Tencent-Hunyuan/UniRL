# Resume prompt — RewardService

重置机器 / 切新 Claude session 后，把下面这段 **"给 Claude 的开场消息"** 复制粘贴作为新 session 的第一条消息，即可让新 Claude 完全进入状态、无需重建上下文。

---

## 你复制这一整段给新 Claude（三 ``` 之间的内容）

```
resume RewardService（在你的本地 checkout 根目录打开）

## 项目状态速查
请先按顺序读这 3 份文件，进入状态再问我要做什么：

1. docs/DEVELOPMENT_LOG.md §17.7 "Resume 入口（覆盖 §16.10，以本节为准）" — 最新状态
2. docs/DEVELOPMENT_LOG.md §17 "跨 repo 调用契约修复（RewardService ↔ DiffusionRL_main）" — 最近一次 session 的完整记录
3. docs/DEVELOPMENT_LOG.md §16 "集成 geneval + ocr + videoalign scorer（两层隔离调查）" — 上一次 session
4. docs/ARCHITECTURE.md — 系统架构全貌（进程拓扑 / 数据流 / 4 层抽象 / 扩展点）

## 当前第一优先级

**统一主线 `integration/geneval-ocr-clean` 的收尾**（已含 geneval+ocr+videoalign + 跨-repo 调用契约修复，两 repo 改动均未 commit）：设 git 身份 → 两 repo 本地 commit（不 push）→ 真机验证 `ocr`（GOT-OCR）+ `videoalign`（VideoReward 权重 + flash-attn venv + 经 `input_kind: video` 的远程 video 端到端）→ 定 `geneval` 长期路线（py3.10 sidecar vs 移植检测栈；py3.13 集群无法经 runtime_env 托管，详见 §16.2 / ARCHITECTURE §5.1.1）。

**调用契约已修复并被契约测试钉死（详见 §17）**：image/video payload 形状、finite-guard（NaN/None/inf→样本失败）、`input_kind` 路由、geneval 缺 metadata raise、videoalign Overall-first，全部 CPU 测试绿。失败处理走 **fail-fast**（用户明确否决"mask 出 advantage"，不要改回去，见 §17.7）。`ocr`/`geneval2` per-item 失败返回 NaN，由 caller 的 finite-guard 翻成失败、fail-fast 停训。
（§15.6 遗留的"多机验证"仍未完成，可一并处理。）

## 工作流约束（必须遵守）

- 每次动代码前用 /code-standards skill 走三段式：项目探索 → 出 plan → 等我批准 → 实施 → simplify/review → 汇报 → 同步 docs/。
  skill 权威版本在 `.claude/skills/code-standards/SKILL.md`（相对仓库根）。父目录同义文件若存在则随后同步。
- 所有 cache 文件必须留在当前目录（.pycache / .pytest_cache / .pip-cache / .install.out），不许写到 /tmp / ~ / $HOME。这是硬约束。
  **例外**：Ray runtime temp-dir（Unix socket / shm / session log）属于进程运行时文件而非缓存，走 /tmp/ray-$USER；详见 docs/DEVELOPMENT_LOG §12.10。
- 仓库内文件引用一律使用相对于仓库根的相对路径；YAML 里的外部资源（权重盘等）仍用绝对路径。
- pytest 命令模板：
  PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
- 装依赖用 ./install.sh（不是 pip install -e .[all]，原因见 §11.10）

## 绝对不要做的事

- 不要把 dtype 默认改回 "auto" — bfloat16 是当前两个 vLLM 模型的实际精度，更显式
- 不要给 build_vllm_llm_kwargs 加"第 13 个具名参数"除非真的有用户在用 — extra_llm_kwargs 就是 escape hatch
- 不要加回 `--ignore-installed` — 会导致 torch 重装，与 base xformers ABI 冲突
- 不要恢复 _compat.py — per-scorer venv 已让每个 scorer 有正确的 transformers
- 不要把 runtime_env 改回可选 — 每个 reward 必须有自己的 venv
- 不要在 base 环境里装 transformers/vllm — 这些是 scorer 级依赖，走 envs/*.txt
- 不要用代码 patch 绕环境问题 — 从 envs/*.txt 版本 pin 解决
- 不要跳过 plan 直接动代码 — 单行修复除外，其他都必须先出 plan
- 不要在仓库内的文档/脚本/skill 里写仓库根的绝对路径 — 用相对路径
- 不要随意调整 configs/service.cluster.example.yaml 里 rewards 的顺序 — 多 GPU actor 必须排最前避免碎片化（§12.15）

## 启动命令

如果是重置后的新机器（在仓库根目录）：
  conda create -n reward-service python=3.12 && conda activate reward-service
  # 先装 torch + nccl（base 环境预装，不走 pip）
  ./install.sh    # 只装 base（ray, fastapi, uvicorn, pillow）

启动服务（单机）：
  PYTHONPATH=. python3.12 -m reward_service --config configs/service.example.yaml
  # 首次启动慢（Ray pip install 每个 scorer 的 venv）；后续复用缓存

启动服务（多机 · 先拉起 Ray cluster，再起 service）：
  export NODE_IP_LIST="ip1:8 ip2:8"
  export HTTP_PROXY=... HTTPS_PROXY=... NO_PROXY=...
  bash scripts/ray_start.sh
  PYTHONPATH=. python3.12 -m reward_service --config configs/service.cluster.example.yaml
  # 停：
  bash scripts/ray_stop.sh

压测：
  PYTHONPATH=. python3.12 scripts/smoke_client.py --url http://localhost:8080
  PYTHONPATH=. python3.12 scripts/bench_concurrent.py --url http://localhost:8080 --sweep 50 100 200 400 800 --total 500

现在告诉我你读完这几份文件后准备怎么推进，或者等我给你新的任务。
```

---

## 文件清单（告诉新 Claude 路径在哪的备忘单）

**Per-scorer venv**：
- `envs/*.txt` — 每个 scorer 的 pip requirements（base/clip/pickscore/imagereward/hpsv2/hpsv3/unified_reward/geneval2/geneval/ocr/videoalign）
- `reward_service/config.py::RewardModelCfg.runtime_env` — 必填字段
- `reward_service/workers/group.py::_build_runtime_env` — 读 requirements → Ray runtime_env dict

**代码**：
- `reward_service/scorers/{unified_reward,geneval2}.py` — vLLM 类 scorer
- `reward_service/scorers/_common.py::build_vllm_llm_kwargs` — vLLM 参数汇总 helper
- `reward_service/config.py` — `ServiceCfg` / `ServerCfg` / `ClusterCfg` / `RewardModelCfg`
- `reward_service/workers/pool.py::_init_ray` — 多机 Ray 接入点
- `reward_service/workers/group.py::_actor_options` — scheduling + runtime_env 透传
- `configs/service.example.yaml` — 单机 8 GPU
- `configs/service.cluster.example.yaml` — 双机 16 GPU
- `scripts/ray_{start,stop,smoke}.sh` + `_ray_lib.sh` — 多机启动/回收/smoke

**测试**：
- §15 时点 131 passed；本 session 新增 test_ocr / test_geneval / test_registry（CPU 上 12 passed / 9 skipped，skip 为无 Levenshtein 的公式测试 + GPU smoke）
- 集成测试：`pytest tests/integration/ -m integration -v`（验证 venv 安装）
- GPU smoke test（`@pytest.mark.gpu + @pytest.mark.slow`）未跑

**文档**：
- `README.md` — 用户手册
- `docs/ARCHITECTURE.md` — 架构稳定概念
- `docs/DEVELOPMENT_LOG.md` — 历史档案（§16 是最新 session，Resume 入口在 §16.8）
- `CHANGELOG.md` — 用户视角变更表
- `docs/RESUME_PROMPT.md` — 本文件

**安装 & 运行**：
- `install.sh` — 精简版：只装 base（server + dev），支持 uv-first
- `pyproject.toml` — base deps only（无 transformers/vllm pin）

## 硬约束清单（供 Claude 参考）

1. 所有 cache 在当前目录，不外溢
2. 动代码先 plan、等批准
3. vLLM 默认 dtype 是 `bfloat16`，不是 `auto`
4. `_compat.py` 已删除——不要恢复
5. `runtime_env` 是必填——不要改回可选
6. base 环境不装 transformers/vllm——走 envs/*.txt
7. 不加回 `--ignore-installed`——会与 base xformers ABI 冲突
8. 不用代码 patch 绕环境问题——从 envs/*.txt 版本 pin 解决
9. 测试文件是 ground truth
8. 仓库内文档/脚本/skill 一律相对路径引用仓库内文件
9. 汇报完必须同步 `docs/`（§5.1 步骤 6）
10. `configs/service.cluster.example.yaml` 里 rewards 顺序不能乱改——多 GPU actor 排最前（§12.15）
