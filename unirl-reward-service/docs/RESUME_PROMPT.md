# Resume prompt — RewardService

重置机器 / 切新 Claude session 后，把下面这段 **"给 Claude 的开场消息"** 复制粘贴作为新 session 的第一条消息，即可让新 Claude 完全进入状态、无需重建上下文。

---

## 你复制这一整段给新 Claude（三 ``` 之间的内容）

```
resume RewardService（在你的本地 checkout 根目录打开）

## 项目状态速查
请先按顺序读这 3 份文件，进入状态再问我要做什么：

1. docs/DEVELOPMENT_LOG.md §19.10 "Resume 入口（覆盖 §19.7，以本节为准）" — 最新状态
2. docs/DEVELOPMENT_LOG.md §19.9 "训练侧接线：新增 videohpsv3 service config + WAN recipe" — 最近一次 session
3. docs/DEVELOPMENT_LOG.md §19.0–19.8 "新增 videohpsv3 T2V reward（对齐上游 Flash-GRPO）" — scorer 本身的完整记录
4. docs/DEVELOPMENT_LOG.md §17 "跨 repo 调用契约修复" + §18 "并入单仓" — 之前两次 session
5. docs/ARCHITECTURE.md — 系统架构全貌（进程拓扑 / 数据流 / 4 层抽象 / 扩展点）

**位置注意**：当前工作副本在 `UniRL` 仓库 worktree `feature/flashgrpo-unirl` 下的 `unirl-reward-service/`。§18.1 提到的 `DiffusionRL_main/RewardService/` 是更早的历史状态，以 §19.10 为准。

## 当前第一优先级

**让 `videohpsv3` 在真机上跑起来**。service 侧（逐帧 HPSv3 + top-30% 聚合，45 tests passed、ruff clean）和训练侧接线（§19.9）都已交付，但**没有任何真机验证**——只做了配置层校验。

已交付的两个新文件（都是新增，原 recipe 零改动）：
- `configs/videohpsv3_service.yaml`
- `examples/diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3.yaml`（相对 UniRL 仓库根）

剩下的事：

1. 下载 HPSv3 权重，填进 `configs/videohpsv3_service.yaml` 的 `weights_path`（现在是占位符）。
2. 腾 GPU（上次 8 卡全被训练占满，需停 PID 1691465 + Ray 集群）。
3. 真机 smoke，按下面"启动命令"里的两步链路。先 `frame_stride: 8` 跑通再决定是否降到 1——**注意这不是"便宜版"而是另一个估计量**：stride 8 时 top-30% 池子只有 3 帧，stride 1 是 24 帧（§19.9.5）。
4. 未决：`wan21_t2v_flashgrpo.yaml`（trainside 变体）要不要也配一个 `_videohpsv3` 兄弟文件；`wan21_t2v_flashgrpo_sglang.yaml` 的 `use_torch_compile: true`。
5. commit 时：`reward_service/scorers/video_hpsv3.py` 和 `tests/` 仍是 untracked，需要 `git add`。

失败处理走 **fail-fast**（用户明确否决"mask 出 advantage"，不要改回去，见 §17.7）。scorer 侧 per-item 失败返回 NaN，由 caller 的 finite-guard 翻成样本失败。
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
- 不要把 `videohpsv3` 的 per-item NaN 改回 raise — 一个坏片段不该拖垮整桶 64 项（§19.6）；但"缺 video / 类型不对"必须继续 raise，别用 NaN 掩盖接线 bug
- 不要在 `video_hpsv3.py` 顶层 `import cv2` — `registry._try_import` 会把 ImportError 咽成 warning，顶层导入会让这个 reward 静默未注册（§19.7）
- 不要给 `videohpsv3` 单独建 `envs/videohpsv3.txt` — 必须与 `envs/hpsv3.txt` 内容一致才能共用 Ray 缓存的 venv
- 不要把 reward 名 `videohpsv3` 改成 `video_hpsv3` — 文件名是 `video_hpsv3.py`，但注册名 / YAML `scorer:` 是线协议标识符，必须与上游 Flash-GRPO 的 config key 一致（§19.8）
- 不要给 `configs/videohpsv3_service.yaml` 加 `cluster:` 段 — 本仓单 reward config 的规范是先 `ray start --head` 再起 service，`editreward_service.yaml` 就没这一段（§19.9.2）
- 不要在 Hydra 启动命令里写光秃秃的 `devices_per_node=7` — 这个 key 不在 recipe 里，会报 `not in struct`，必须 `++devices_per_node=7`（§19.9.5）
- 不要把 client `timeout` 降到 ≤ server `score_timeout_s` — 客户端会先放弃并按 `max_retries: 3` 把整批重 POST，把已经在烧的 GPU 时间乘以 3（§19.9.5）
- 不要去"修" 视频轨道 `[T,C,H,W]` 与 `_encode_video_b64` 要的 `(C,T,H,W)` 的差异 — `unirl/types/reward.py:73-77` 的 permute 已经桥接了（§19.9.4）
- 不要改原来那两个 flashgrpo recipe 去接 remote reward — 用户明确要求新增文件（§19.9.1）

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

FlashGRPO videohpsv3 链路（两步，§19.9）：
  # 节点 1（reward）
  ray start --head --port=6379 --num-gpus=1
  python -m reward_service --config configs/videohpsv3_service.yaml

  # 节点 2（train，在 UniRL 仓库根）
  export REWARD_SERVICE_URL=http://<node1_ip>:8080
  bash examples/run_experiment_single_node.sh \
    diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3

  # 单机共卡：必须留一张卡给 scorer actor，两个 override 都要（第二个必须 ++）
  bash examples/run_experiment_single_node.sh \
    diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3 \
    num_devices=7 ++devices_per_node=7

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
- `configs/videohpsv3_service.yaml` — FlashGRPO WAN T2V 专用（单 reward、1 GPU），消费方是 `examples/diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3.yaml`（相对 UniRL 仓库根）
- `scripts/ray_{start,stop,smoke}.sh` + `_ray_lib.sh` — 多机启动/回收/smoke

**测试**：
- §15 时点 131 passed；本 session 新增 test_ocr / test_geneval / test_registry（CPU 上 12 passed / 9 skipped，skip 为无 Levenshtein 的公式测试 + GPU smoke）
- 集成测试：`pytest tests/integration/ -m integration -v`（验证 venv 安装）
- GPU smoke test（`@pytest.mark.gpu + @pytest.mark.slow`）未跑

**文档**：
- `README.md` — 用户手册
- `docs/ARCHITECTURE.md` — 架构稳定概念
- `docs/DEVELOPMENT_LOG.md` — 历史档案（§19 是最新 session，Resume 入口在 §19.7）
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
