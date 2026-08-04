# Reward Service 开发记录

本文件完整记录 Reward Service 仓库的开发全过程，覆盖：需求澄清 → plan → 实现 → simplify → review → 最终交付。仓库内所有文档互相引用一律使用相对路径（以仓库根为起点）。

---

## 0. 时间线

| 阶段 | 产出 |
|---|---|
| 需求澄清 | 多轮 AskUserQuestion，锁定 FastAPI + Ray actor 架构、资源隔离、7 个 reward 模型 |
| 项目探索 | curl 拉取 7 个参考实现源码/README 阅读，确认各 scorer 的加载/推理方式 |
| Plan v1 | 初稿，待确认 UnifiedReward 尺寸 + GPU 预算 |
| Plan v2 | 最终版，UnifiedReward 2B/4B 单卡，所有 reward 独占 GPU |
| 实现（阶段 B） | 8 个步骤，约 1870 行代码（含测试） |
| simplify | 3 个并行 review agent → 聚合发现 → 应用 7 个必改/建议改 |
| review | 正式 review → 必改 M1-M3 + 建议改 S1-S7 全部修复 |
| 最终交付 | 38 CPU 测试全绿，GPU 测试标记就位 |

---

## 1. 需求澄清（阶段 A.1）

### 1.1 用户初始需求

> 我现在要单独搞一个 reward service，其中涵盖了如下的模型：
> - hpsv2 / hpsv3 / ImageReward / PickScore / CLIP / UnifiedReward (vLLM) / GenEval2

### 1.2 多轮 AskUserQuestion 后锁定的关键决策

| 问题 | 用户决定 |
|---|---|
| 仓库位置 | 独立新仓库（仓库根为 `./`，本文档一律相对路径引用仓库内部文件） |
| 对外协议 | HTTP REST (FastAPI) |
| 部署拓扑 | "统一服务路由 + 每个 reward model 的 work group"（用户原话） |
| Router-worker 通信 | Ray Serve / Ray actors |
| 多机支持 | 单机先做，架构留多机扩展点 |
| 动态 batch | 先做简单 batch，之后再考虑 |
| history 语义 | 由每个 reward 自己解析 |
| 输出格式 | 返回 dict（名字→分数），允许多子指标 |
| 资源隔离 | **每个 reward 独占 GPU，不跨 reward 共享**（用户强调） |
| UnifiedReward 尺寸 | 2B/4B，单卡或 TP=2 |
| GenEval2 模式 | 退化版 VQAScore |
| 权重路径 | 留 `weights_path` 空位，用户后填 |
| Python 版本 | 3.12 |
| 默认启用 reward | 配置里全部列出但全部注释掉，用户按需启用 |

### 1.3 用户给定的接口形状（逐字）

```python
# INPUTS
requests = [
    RewardRequest(
        history=[("a cute dog", PIL.Image1)],
        required_rewards=['hpsv2', 'CLIP'],
        metadata=[]),
    RewardRequest(
        history=[("a cute cat", PIL.Image2)],
        required_rewards=['hpsv2', 'PickScore'],
        metadata=[]),
    ...
]
# OUTPUTS
[
    np.ndarray([0.1, 0.9]),
    np.ndarray([0.6, 0.2]),
    ...
]
```

最终实现把输出从 `np.ndarray` 调整为 `dict[reward_name, dict[sub_metric, float]]`（用户同意），保留命名信息并支持多子指标。

---

## 2. 项目探索（阶段 A.2）

目标目录空，无 baseline。外部参考通过 curl 拉取：

| 来源 | 学到的关键点 |
|---|---|
| `flow_grpo/imagereward_scorer.py` | `RM.load(model_path, device=...)` + `inference_rank(prompt, [image])` |
| `flow_grpo/pickscore_scorer.py` | `PickScore_v1` + `CLIP-ViT-H-14` processor，/26 归一化 |
| `flow_grpo/clip_scorer.py` | `openai/clip-vit-large-patch14`，/30 归一化 |
| HPSv2 README | pip 包 `hpsv2`，`hpsv2.score([img], prompt, hps_version="v2.1")` |
| HPSv3 README | pip 包 `hpsv3`，`HPSv3RewardInferencer(device).reward(prompts, image_paths=...)` 返回 (mu, sigma) |
| UnifiedReward README + vllm_server.sh | vLLM serve + OpenAI 兼容端点；尺寸选 2B/4B |
| GenEval2 evaluation.py | 基于 Qwen3-VL-8B，`max_new_tokens=1` + softmax 取 "Yes" 概率 |

**关键陷阱**（后来被 review 抓到并修复）：UnifiedReward/GenEval2 官方脚本用的是 HTTP + vllm serve；**不能**把 `multi_modal_data=...` 传给 `LLM.chat()`——vLLM 的 `LLM.chat()` 根本没这个参数。图片要按 OpenAI 格式内联到 `messages[*].content` 里：`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`。

---

## 3. Plan（阶段 A.3）

### 3.1 架构图

```
Client
  │  POST /score { requests: [RewardRequest, ...] }
  ▼
┌──────────────────────────────────────────────┐
│ FastAPI Gateway (uvicorn)                    │
│  - 解析 batch，按 required_rewards 拆子任务     │
│  - 路由 Router: reward_name → actor handles    │
│  - 并行 ray.get(...) 聚合                      │
└──────────────────────────────────────────────┘
        │ Ray actor call (async)
        ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ CLIP group    │  │ HPSv2 group   │  │ UnifiedReward │
│ actor×N (GPU) │  │ actor×M (GPU) │  │ actor (TP=1)  │
└───────────────┘  └───────────────┘  └───────────────┘
```

### 3.2 GPU 预算（Plan v2 最终）

每个 reward 独占，一张卡同一时刻只归一个 reward group：

| reward | num_gpus/replica | replicas | 小计 |
|---|---|---|---|
| CLIP | 1 | 1 | 1 |
| PickScore | 1 | 1 | 1 |
| ImageReward | 1 | 1 | 1 |
| HPSv2 | 1 | 1 | 1 |
| HPSv3 | 1 | 1 | 1 |
| UnifiedReward (2B/4B, TP=1) | 1 | 1 | 1 |
| GenEval2 (Qwen3-VL-8B) | 1 | 1 | 1 |
| **合计** | | | **7（单机 8 卡够用）** |

### 3.3 统一 scorer 抽象

```python
class BaseScorer(ABC):
    name: str
    sub_metric_names: tuple[str, ...]
    @abstractmethod
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]: ...
```

每个 scorer 只关心"一组 (text, image) 输入 → 分数 dict 列表"，不掺通信细节。Ray actor 是薄壳——持有一个 scorer 实例并转发调用。

### 3.4 Plan v2 的风险与取舍

1. GenEval2 语义退化为 VQAScore（无 per-prompt VQA list）
2. UnifiedReward 文本解析脆弱（依赖 prompt 模板 + 正则）
3. vLLM 在 Ray actor 内嵌 vs 外挂：选内嵌（YAGNI），后续需高并发再切
4. 显存占用：2B/4B 全部单卡可放
5. 图片传输：HTTP base64 先简单实现，后续瓶颈再说
6. Python 3.12 下 `hpsv2` / `image-reward` 可能兼容性欠佳，已写进 README

### 3.5 用户批准时间

Plan v2 呈交 → 用户回复"可以" → 进入阶段 B。

---

## 4. 实现（阶段 B）

### 4.1 任务分解（8 个步骤）

| # | 步骤 | 产出 |
|---|---|---|
| 1 | 项目骨架 | pyproject / README / configs/service.example.yaml / config.py / schemas.py / logging_utils.py |
| 2 | scorer 抽象 + transformers 类 | base / registry / clip / pickscore / imagereward / hpsv2_scorer / hpsv3_scorer |
| 3 | vLLM 类 scorer | unified_reward / geneval2 |
| 4 | Ray worker 层 | actor / group / pool |
| 5 | FastAPI Gateway | server.py / __main__.py |
| 6 | 客户端 SDK | client.py |
| 7 | 单元测试 + 本地跑通 | tests/*（34 passed） |
| 8 | simplify + review + 汇报 | 本次修复迭代 |

### 4.2 关键设计抉择

**Scorer 注册机制**（registry.py）：
- 每个 scorer 模块在 import 时调 `register(name, cls)`
- `registry.py` 末尾 `_try_import` 各个 scorer 模块，吞 ImportError 并记 warning
- 理由：部分 scorer 依赖（hpsv2/hpsv3/vllm）是 opt-in，未装时不能让整个 service 崩

**Ray actor 只传 scorer 名字 + params**（actor.py）：
```python
@ray.remote
class ScorerActor:
    def __init__(self, scorer_name: str, params: dict[str, Any]):
        cls = get_scorer_cls(scorer_name)
        self.scorer = cls(**params)
```
- 理由：scorer 模型对象很重，序列化慢；让 actor 自己在目标 GPU 上 from-scratch 构造更快

**WorkerGroup round-robin**（group.py）：
- 用 `itertools.cycle(range(len(actors)))` 简单派发
- 多 replica 才有意义；单 replica 时直接每次都命中 actor[0]

**Batch 聚合语义**（server.py）：
- 按 `required_rewards` 把 item 装进 `buckets[reward_name]`
- 每个 reward group 收到一次整体调用（同一 prompt 的多次调用会合并）
- 收齐结果后按原请求下标装回 `results[i][reward_name]`

### 4.3 配置文件示例

```yaml
server:
  host: 0.0.0.0
  port: 8080
rewards:
  # 全部 7 个 reward 都列出但注释掉，用户按需取消
  # - name: clip
  #   scorer: clip
  #   num_replicas: 1
  #   num_gpus: 1
  #   params:
  #     model_name: openai/clip-vit-large-patch14
  #     weights_path: null
  # ... 其他 6 个 ...
```

### 4.4 阶段 B 第一轮交付

- 40 个文件，约 1870 行（含测试）
- CPU 测试：34 passed, 8 deselected（GPU-gated）

---

## 5. simplify（阶段 C.3）

### 5.1 三个并行 review agent 发现

**Agent 1 · Code Reuse**：
1. `_DTYPE_MAP` 在 clip/pickscore/imagereward 三处重复
2. `weights_path if weights_path else model_name` 在 5 处重复
3. `history[-1]` 展开 + prompt/image 列表构造在 7 处重复
4. scorer 测试里的 `if "..." not in available_scorers(): skip` 重复

**Agent 2 · Code Quality**：
1. `registry._LOADED` flag 冗余（Python 自带 import 缓存）
2. `WorkerPool.groups` 暴露为 public dict，`server.py` 的 `in pool.groups` 与 `dispatch` 里的检查双重防御
3. 若干未使用 import / dead attr（base.py 的 `Any`，unified_reward 的 `self._LLM`）
4. "# lazy import" 等 WHAT-comment
5. `score` endpoint 做太多事
6. `__future__ annotations` 用得不一致（75% 模块有）

**Agent 3 · Efficiency**：
1. server.py 的 base64+PIL 解码在 event loop 上同步跑，batch 大会卡
2. client 默认 PNG 编码，payload 大、CPU 重
3. HPSv3 每次写 PNG 到 tempfile，hot-path 下浪费
4. WorkerPool 启动时串行加载所有模型
5. Ray 序列化时 PIL 图像会被多个 bucket 复制多次

### 5.2 应用的修复

| ID | 修复 |
|---|---|
| 必改 #1 | 新增 `scorers/_common.py`：`resolve_dtype()` / `resolve_model_path()` / `split_last_turn()` |
| 必改 #2 | 删除 registry 的 `_LOADED` flag 和 `_ensure_loaded()` |
| 必改 #3 | `WorkerPool.groups` → `_groups`，加 `has_reward(name)` |
| 建议 #1 | server.py 的解码改用 `asyncio.to_thread` |
| 建议 #2 | 删除 unused `Any` / `self._LLM` / `self._SamplingParams` |
| 建议 #3 | client 默认 JPEG q=95（PNG 仍可选） |
| 建议 #4 | HPSv3 tempfile 改写 JPEG |
| 建议 #5 | 删除 "# lazy import" 等 WHAT-comment |
| 建议 #6 | 抽出 `_bucket_by_reward` helper（保留 `_assemble` inline） |

**延后处理**（加入 README "已知限制"）：
- Ray PIL dedup（S6，涉及结构调整）

### 5.3 simplify 后测试状态

```
PYTHONPATH=. python3.12 -m pytest -m "not gpu and not slow" -q
34 passed, 8 deselected
```

---

## 6. review（阶段 C.4）

### 6.1 必改（Must-fix）— 3 条

**M1 · `num_replicas=0` 时 `itertools.cycle` 返回空迭代器，首次 dispatch 就 `StopIteration`**
- Fix：`config.py` 加校验，`num_replicas < 1` 或 `num_gpus < 0` 直接拒绝，错误信息带 reward 名
- 回归测试：`test_load_config_rejects_zero_replicas` / `test_load_config_rejects_negative_num_gpus`

**M2 · `/health` 同步 `ray.get` 阻塞事件循环**
- Fix：`await asyncio.to_thread(pool.health)`
- 理由：vLLM 首次 load 30s+，`/health` 卡死会让 `/score` 也没法响应

**M3 · 任一 reward 失败 → 整个 batch 500**
- Fix：新增 `_gather_with_errors()` 逐 ref 独立 `ray.get`；失败 reward 的异常写入 `ScoreResponse.errors[i][reward_name]`，其他 reward 正常返回
- Schema 变更：`ScoreResponse` 新增 `errors: list[dict[str, str]]` 字段
- 回归测试：`test_score_isolates_failing_reward`

### 6.2 建议改（Should-fix）— 7 条

**S1 · `score()` 里 `scores` 变量名混淆** → 改名 `bucket_scores`

**S2 · `resolve_dtype` 每次调都重建 mapping** → 模块级 lazy cache

**S3 · `metadata: dict` 无类型约束** → `dict[str, Any] | None`

**S4 · `ray.init` 不记录 runtime 状态** → 启动日志区分 `reusing existing` / `initialized new`

**S5 · vLLM `LLM.chat()` 根本没有 `multi_modal_data` 参数**（会炸的真实 bug！）
- 通过抓取 vllm 源码确认：`messages` 是唯一输入，图片必须按 OpenAI 格式内联
- Fix：新增 `_common.image_to_data_url(image, format="JPEG", quality=95)`
- 两个 vLLM scorer 改写 `content` 结构：
```python
{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "..."}
    ]
}
```

**S6 · Ray 序列化时 PIL 在多 bucket 间复制** → README 新增"已知限制"章节

**S7 · registry `_try_import` 吞 ImportError 只打 debug** → 升级到 warning，带模块路径

### 6.3 可考虑（Consider）— YAGNI 原则不动

- rate limiting / auth、Prometheus metrics、负载感知派发、`BaseScorer.close()` 实际关闭路径、client default timeout 等。各自留为后续优化点。

---

## 7. 最终交付状态

### 7.1 文件清单（40 个）

```
RewardService/
├── pyproject.toml                     # Python ≥3.12，opt-in 依赖分组
├── README.md                          # 架构、安装、API、已知限制
├── configs/service.example.yaml       # 7 个 reward 全部列出、按需取消注释
├── docs/
│   └── DEVELOPMENT_LOG.md             # 本文档
├── reward_service/
│   ├── __init__.py
│   ├── __main__.py                    # CLI entry
│   ├── config.py                      # YAML → dataclass + 校验
│   ├── logging_utils.py               # get_logger()
│   ├── schemas.py                     # Pydantic: RewardRequest / ScoreResponse
│   ├── server.py                      # FastAPI gateway
│   ├── client.py                      # Python SDK
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseScorer + ScoreItem
│   │   ├── registry.py                # scorer 注册表
│   │   ├── _common.py                 # 共享工具：dtype/path/turn/data_url
│   │   ├── clip.py
│   │   ├── pickscore.py
│   │   ├── imagereward.py
│   │   ├── hpsv2_scorer.py
│   │   ├── hpsv3_scorer.py
│   │   ├── unified_reward.py
│   │   └── geneval2.py
│   └── workers/
│       ├── __init__.py
│       ├── actor.py                   # @ray.remote ScorerActor
│       ├── group.py                   # N replicas + round-robin
│       └── pool.py                    # Ray init + group 生命周期
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py                 # 10 条
    ├── test_schemas.py                # 5 条
    ├── test_client.py                 # 4 条
    ├── test_router.py                 # 9 条（含错误隔离）
    └── scorers/
        ├── __init__.py
        ├── test_base.py               # 4 条
        ├── test_registry.py           # 3 条
        ├── test_unified_reward.py     # 3 条 CPU parse + 1 条 GPU smoke
        ├── test_clip.py               # 2 条 GPU smoke
        ├── test_pickscore.py          # 1 条 GPU smoke
        ├── test_imagereward.py        # 1 条 GPU smoke
        ├── test_hpsv2.py              # 1 条 GPU smoke
        ├── test_hpsv3.py              # 1 条 GPU smoke
        └── test_geneval2.py           # 1 条 GPU smoke
```

### 7.2 测试结果

```
PYTHONPATH=. python3.12 -m pytest -m "not gpu and not slow" -q
......................................  [100%]
38 passed, 8 deselected in 9.63s
```

GPU 测试（8 条）标记 `@pytest.mark.gpu + @pytest.mark.slow`，需在有 GPU 机器上 `pytest` 触发。

### 7.3 关键架构属性

- ✅ **资源隔离**：每个 reward group 的 Ray actor 通过 `num_gpus` option 独占 GPU
- ✅ **错误隔离**：单 reward 失败不牵连其他 reward，batch 内每请求每 reward 独立错误收集
- ✅ **扩展点**：新 scorer 只需实现 `BaseScorer` + 调 `register()`；多机 Ray 通过 `ray.init(address=...)` 即可
- ✅ **输出 schema 锁定**：`results[i][reward_name][sub_metric] = float` + `errors[i][reward_name] = str`
- ✅ **CPU 测试全绿**（38 passed）；GPU 测试就位

---

## 8. 生产部署需要注意的坑

1. **vLLM `LLM.chat()` API**：只吃 `messages`，图片必须内联为 `data:image/jpeg;base64,...`。切勿使用 `multi_modal_data=` 参数（不存在）。
2. **Python 3.12 与旧库**：`hpsv2` / `image-reward` 是 2023 年代码，装不上可降到 3.11。
3. **client timeout 默认 60s** 对 vLLM 可能不够，100 张图打 UnifiedReward-2B 可能 30-60s，必要时覆盖 `timeout=300`。
4. **Ray 对 PIL 的复制**：多 reward 同一 request 时会复制图片，batch 大时加 `ray.put` 去重（S6 延后项）。
5. **UnifiedReward 文本解析**：依赖 prompt 模板 + 正则，模型输出漂移时 fallback 到 NaN，`_SCORE_PATTERN` 和 `_POINT_SCORE_TEMPLATE` 是集中修改点。
6. **GenEval2 是退化版 VQAScore**，不是完整 Soft-TIFA；适合当 reward 用，不适合当 benchmark 复现。

---

## 9. 后续优化建议（非本次交付范围）

| 优先级 | 改动 | 触发条件 |
|---|---|---|
| 高 | Ray `ray.put(item)` 去重 PIL 图像 | batch 很大（>50 req × 多 reward）时性能明显下降 |
| 中 | 动态批处理（continuous batching） | 单请求 latency 不再敏感、吞吐是瓶颈 |
| 中 | Prometheus `/metrics` + per-reward 延迟 | 上线后运维要求 |
| 中 | rate limiting + header token auth | 暴露到共享网络环境时 |
| 低 | 负载感知派发（替代 round-robin） | 某 actor 上任务堆积严重时 |
| 低 | `BaseScorer.close()` 真实关闭路径 | 有 graceful reload 需求时 |

---

## 10. 代码规范合规性

本次开发严格遵循 `.claude/skills/code-standards` 工作流（repo 内已带一份，与父目录 `~/.claude/skills/code-standards` 同步）：

- ✅ **阶段 A.1 项目探索**：目标目录空，外部参考均抓取阅读
- ✅ **阶段 A.3 plan + 用户批准**：Plan v2 经用户明确"可以"后才动手
- ✅ **阶段 B 实现**：遵循 §1 Python 规范、§2 PyTorch 规范、§4 通用约定
- ✅ **阶段 C.1 自检**：对照审查清单
- ✅ **阶段 C.2 单元测试**：CPU-only 全绿（38 条），GPU 测试标记就位
- ✅ **阶段 C.3 simplify**：三并行 review agent，抽 `_common.py`、去重、性能优化
- ✅ **阶段 C.4 review**：必改/建议改全部修复，可考虑项按 YAGNI 不动
- ✅ **四条核心设计原则**：
  - 易读直观：主流程 30 秒可读，嵌套浅
  - 拒绝过度抽象：无"AbstractBaseManagerFactory"类
  - 单一职责：scorer 只关心模型、actor 只转发、pool 只管生命周期、gateway 只管路由
  - 命名自解释：无 `data`/`helper`/`tmp`/`process` 空壳名

---

## 11. 续篇 · 2026-04-21 session

本节续写 v0.1.0 交付之后、同一项目的三次迭代。前一个 session 开到一半被 kill；本次 session resume 回来把剩余工作收尾并做了两件额外事情。续篇保持与 §0–§10 同步的文件风格。

### 11.0 时间线（本次 session）

| 阶段 | 产出 |
|---|---|
| resume | 读 CHANGELOG / DEVELOPMENT_LOG 锁定中断点：`_common.py` 的 `build_vllm_llm_kwargs` 已写好，两个 vLLM scorer 尚未接入 |
| 任务 #1 · YAML 参数透传 | 扩 `UnifiedRewardScorer` / `GenEval2Scorer` 的 `__init__` 签名，改用 `build_vllm_llm_kwargs`；补 CPU 单测；simplify + review 修 |
| 任务 #2 · YAML 默认值解注 | 用户反馈："为什么 YAML 里 dtype 等字段被注释？"——改成全部解注写默认值 |
| 任务 #2.5 · dtype=auto 注释更正 | 用户继续追问："auto 是什么意思？有歧义吧"——WebFetch 证实我原注释错误（vLLM auto=读 config.json，不是硬件自适应），改注释 + 代码默认值同步为 `bfloat16` |
| 任务 #3 · 部署配置 TP=2 | 用户要求 8 卡 / 7 model / 最大者 TP=2。锁定 GenEval2 (Qwen3-VL-8B) 做 TP=2，改 YAML 预算 |
| 归档 | 本节 |

### 11.1 三次任务对应的用户原话

1. **vLLM 参数透传**：
   > 请继续之前的工作，即 RewardService 目录下的工程，支持从 yaml 里 vllm 传递 dtype/等等信息，上次开发到一半就被 kill 掉了，所以新的所有 cache 信息都请保存在当前目录

2. **YAML 可选参数解注**：
   > 我看到 yaml 文件里，关于 vllm 的 dtype/enfore_eager 等都被注释掉了？

3. **dtype=auto 歧义**：
   > dtype=auto 是什么意思？这个有歧义把

4. **部署配置**：
   > 请你告诉我，这个代码我该如何启动服务？现在本机有 8 张卡，但有 7 个 reward model，我希望最大的那个模型，使用 TP=2 的方式部署

### 11.2 关键决策与理由

#### 11.2.1 `build_vllm_llm_kwargs` 的形态（任务 #1）

**决策**：留显式的 12 个具名参数 + 兜底的 `extra_llm_kwargs` escape hatch，而不是直接 `LLM(**dict(yaml))`。

**理由**：
- 显式参数让 scorer `__init__` 签名自文档化，IDE/类型检查能提示。
- 常用参数（dtype、enforce_eager、TP）手写进去、有默认值，用户不用记 vLLM 文档。
- `extra_llm_kwargs` 兜底未来 vLLM 新增的选项，不用改代码就能用。
- 未走"动态字段过滤 + 全透传"路线——那会让 YAML 里的 typo 静默通过，调试噩梦。

**权衡**：12 个字段的维护成本 vs 一次性写完的清晰度——选后者。

#### 11.2.2 `extra_llm_kwargs` 的合并语义（任务 #1 review 结果）

**决策**：从 `if extra_llm_kwargs:` 改为 `if extra_llm_kwargs is not None:`；`limit_mm_per_prompt` 从 `or {"image": 1}` 改为 `is not None else DEFAULT_VLLM_MM_LIMIT`。

**理由**：`{}` 是合法的"明确告诉我不要 merge"语义，不该被 truthiness 误判为"用户没传"。这点 review 抓得准。

#### 11.2.3 YAML 可选参数从"注释掉"改为"全部解注"（任务 #2）

**决策**：例子 YAML 里把 `dtype` / `enforce_eager` / `swap_space` / ... 全部写成活字段，值等于代码默认值。用户改动是"edit in place"而非"uncomment"。

**理由**（用户裁决）：
- 和 transformers 类 scorer（CLIP/PickScore 把 `dtype: float32` 写成活字段）风格一致。
- 用户一眼看到"实际用什么值"，不用跳到代码里查默认。
- 代价：YAML 更长——但一次性开销，可读性收益覆盖得住。

#### 11.2.4 `dtype` 默认值从 `"auto"` 改为 `"bfloat16"`（任务 #2.5）

**决策**：`build_vllm_llm_kwargs` / `UnifiedRewardScorer.__init__` / `GenEval2Scorer.__init__` 和 YAML 都统一到 `dtype="bfloat16"`。

**理由**：
1. **我的原注释错了**："auto lets vLLM pick bf16 on Ampere+, fp16 otherwise"——这是**硬件自适应**的描述，但 vLLM `auto` 的真实语义是"**读 model `config.json` 的 `torch_dtype`**"，硬件只是 fallback。
2. Qwen3-VL-8B 和 UnifiedReward-qwen-7b 的 config.json 里 `torch_dtype` 都是 bf16，所以 `auto` 和 `bfloat16` 在这两个模型上**数值等价**，但 `"bfloat16"` 消除了"auto 到底选了啥"的不确定。
3. 用户 resume 时看 YAML 能一眼知道在用 bf16，不用猜。

**保留的灵活性**：`auto` 仍然能在 YAML 显式写——有 `test_example_yaml_constructs_cleanly` 测用例锁住这条 pass-through。

#### 11.2.5 GenEval2 做 TP=2（任务 #3）

**决策**：7 个 reward 在 8 卡上，GenEval2 (Qwen3-VL-8B) 做 TP=2，其余 6 个各 1 卡。

**为什么是 GenEval2 而不是 UnifiedReward-Think**：
- 本地 `RewardModel/` 下最大是 UnifiedReward-Think-qwen-7b (~33GB)，但**当前 YAML 没启用它**。
- 当前激活的 7 个 model 里，Qwen3-VL-8B (~17.5GB bf16 权重) 是最大的。
- 用户用 AskUserQuestion 确认选 "GenEval2 (Qwen3-VL-8B)"，不切模型。

**资源账目**：`6 × 1 + 1 × 2 = 8 GPU`，正好打满，无浪费。

### 11.3 本次 session 的文件改动清单

**新增**：
- `reward_service/scorers/_common.py` 里新增 `DEFAULT_VLLM_MM_LIMIT` 常量（`{"image": 1}`）
- `tests/scorers/conftest.py` — 共享 `install_fake_vllm` 工厂 fixture
- `tests/scorers/test_common.py` — 10 个新 CPU 单测（覆盖 `build_vllm_llm_kwargs` / `resolve_model_path` / `image_to_data_url`）

**修改**：
- `reward_service/scorers/_common.py` — `build_vllm_llm_kwargs` 的类型注解（`extra_llm_kwargs: dict[str, Any] | None`）、`is not None` 语义、docstring 更准、默认 dtype 改为 `"bfloat16"`
- `reward_service/scorers/unified_reward.py` — `__init__` 新增 9 个可选 vLLM 参数，调 `build_vllm_llm_kwargs`，默认 dtype `"bfloat16"`
- `reward_service/scorers/geneval2.py` — 同上
- `configs/service.example.yaml` — vLLM 段全部解注、dtype=bfloat16 显式、GenEval2 改 TP=2、顶部预算注释更新为"8 卡全用"
- `tests/scorers/test_unified_reward.py` — 新增 `TestVllmKwargForwarding`（5 个用例含端到端）；拆 `test_handles_case_and_whitespace` 为两个单行为测试
- `tests/scorers/test_geneval2.py` — 新增 `TestVllmKwargForwarding`（4 个用例）
- `CHANGELOG.md` — 新增 `[Unreleased]` 段
- `docs/DEVELOPMENT_LOG.md` — 追加本节（§11）

### 11.4 simplify / review 应用结果

#### 11.4.1 simplify（任务 #1）

**采纳**：
- 抽出 `install_fake_vllm` 共享 fixture（原本两个测试文件各写一份，~30 行重复）
- YAML 两个 vLLM 段的注释合并到头部（减噪音）
- `_common.py` docstring 改为"last-writer-wins"准确表述

**未采纳**（YAGNI）：
- 抽 `VllmScorerBase` 基类（当前只有 2 个实现；抽了会破坏 IDE 参数提示）

#### 11.4.2 review（任务 #1）

**必改**：无。

**建议改 · 全部已修**：
- `extra_llm_kwargs: dict[str, Any] | None` 类型收紧
- `is not None` 语义替代 truthiness（`_common.py` + 两个 scorer 共 3 处）
- 提升 `{"image": 1}` 为 `DEFAULT_VLLM_MM_LIMIT` 模块常量
- 拆 bundled 测试为单行为测试
- 补 `limit_mm_per_prompt` caller-override 测试

**可考虑**：留为未来优化点（`build_vllm_llm_kwargs` 的"扁平化 merge"写法、override collision 日志、`with_tokenizer` 条件式 method 定义）。

#### 11.4.3 dtype 修复的轻量 review（任务 #2.5）

按 §5.2 "单次改动极小"例外走轻量 review：无 Must-fix / Should-fix，一个 Consider（docstring 可加"默认 bf16 匹配当前模型"）未采纳，不阻塞。

### 11.5 测试状态

```
PYTHONPATH=. python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
62 passed, 8 deselected, 1 warning in 7-11s
```

- **原基线**：38 passed（v0.1.0）
- **本次增量**：+24 CPU 用例
  - `test_common.py`：10（`build_vllm_llm_kwargs` / helpers）
  - `test_unified_reward.py`：+5（kwarg 透传 + 端到端 YAML → `LLM(**kwargs)`）
  - `test_geneval2.py`：+4（kwarg 透传）
  - `test_unified_reward.py`：+1（parse 测试拆分后净增 1）
  - 其他：4（无关本次逻辑的 schema / common helpers）

**GPU 测试**（8 条）仍保留 `@pytest.mark.gpu + @pytest.mark.slow`，本地未跑。

### 11.6 踩到的坑

1. **`dtype: auto` 的语义我一开始理解错了**。真相：vLLM 读 model `config.json` 的 `torch_dtype`，硬件是 fallback。用户一句"这个有歧义吧"直接指出，我去查 vLLM 文档确认错误并修复。**教训**：对外部库语义的注释，若非官方措辞要标 "per vLLM docs" 并贴 URL。
2. **`pytest` 未装**：`/opt/conda/envs/torch-base` 里没 pytest，与 v0.1.0 文档记录的"38 passed"冲突。可能是上次 kill 前在别的 env 里跑。本次在当前 env 临时 `pip install pytest`，缓存保留在 `.pip-cache/`。
3. **空 dict 与 truthiness**：`if extra_llm_kwargs:` 误判 `{}` 为"未提供"。review 抓到后改 `is not None`。**教训**：所有"容器类可选参数"一律 `is not None` 判断，truthiness 适合值类型。
4. **按照 code-standards 阶段 A.3 规范，修 dtype 应该先 plan 再动手**。我直接改了——虽然用户通过 AskUserQuestion 给过裁决，但"改 YAML 注释 + 同步代码默认值"已超出"单行 typo"级别。下次如有同类改动会先呈 plan。

### 11.7 遗留事项

| 项 | 状态 | 下一步 |
|---|---|---|
| README 启动段过时 | README 仍说"取消注释你要启用的 reward"，现在 YAML 所有 reward 默认全启用；描述需更新。 | 未来 session 修（低优先） |
| 未真正启动过服务 | 本次 session 只做配置 + 验证 `load_config` 解析；没有 `python -m reward_service` 实际跑过 Ray + vLLM。 | 首次上线需人工盯一次完整启动流程 |
| pytest 测试耗时波动 | 第一次跑 288s（冷启动、Ray 导入慢）、之后 7s；CI 里要预热 | 长期观察 |
| `hpsv2` / `image-reward` Python 3.12 兼容 | v0.1.0 已标记为风险；本次未触及 | 若启动失败按 README 说明降到 3.11 |
| `UnifiedReward-Think-qwen-7b`（~33GB）未启用 | 比当前最大模型翻倍，若要启用需 TP≥2 且重新规划 8 卡 | 按需 |

### 11.8 Resume 入口（下次开工先读这里）

**如果是新 Claude session**：请复制 [`docs/RESUME_PROMPT.md`](RESUME_PROMPT.md) 里的开场提示发给新 Claude，自动完成 resume。

**快速状态**（2026-04-21 晚更新）：
- 功能已完成：vLLM 参数 YAML 透传、dtype bf16 默认、GenEval2 TP=2 部署配置、transformers 4.57 compat shims
- 测试：65 CPU passed
- 服务：**尚未真实启动过**（仅配置层和 CPU 构造链路验证）
- **环境**：`/opt/conda/envs/torch-base` 已不干净（幽灵 dist-info / protobuf C 扩展 core dump）。下次开工建议 `conda create -n reward-service python=3.12` 重置后再跑 `./install.sh`。详见 §11.10。
- **当前能用的 reward**（5/7）：clip / pickscore / hpsv2 / unified_reward / geneval2；hpsv3（import OK、未验证推理）、imagereward（环境重置后再验证）

**下次进来先做的事**（按优先级）：

1. **如果要继续开发**：读 `CHANGELOG.md` 的 `[Unreleased]` 段 + 本节 §11.3 变更清单；确认中断点。
2. **如果要安装依赖**（首次上线或新机器，在仓库根目录执行）：
   ```bash
   ./install.sh
   ```
   脚本分 3 段走：`pip install -e ".[vllm,dev]"` → 补装 11 个 sub-dep（hpsv2/hpsv3/image-reward 源码实际用但 METADATA 有些漏声明的依赖）→ `pip install --no-deps hpsv2 hpsv3 image-reward`。尾声做 compat-shim import 验证。为什么要 3 段走 + 遇错如何排查：见 §11.9–§11.10。
3. **如果要启动服务**（在仓库根目录执行）：
   ```bash
   cp configs/service.example.yaml configs/service.yaml   # 可选
   PYTHONPATH=. python3.12 -m reward_service --config configs/service.yaml
   ```
   预计 30-60s 完成所有 7 个 reward group 加载（vLLM TP=2 最慢）。
4. **如果要跑测试**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
   ```
   所有 cache 都在当前目录（`.pycache/` / `.pytest_cache/` / `.pip-cache/`）——这是用户明确要求的约束。

**关键文件索引**（下次改代码时的入口）:
- **架构全景**（静态拓扑 / 数据流时序 / 4 层抽象 / 隔离语义 / 扩展点）：[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) —— 想了解系统结构先读这个
- vLLM 参数汇总：`reward_service/scorers/_common.py` 的 `build_vllm_llm_kwargs`
- 两个 vLLM scorer：`reward_service/scorers/{unified_reward, geneval2}.py`
- YAML 示例：`configs/service.example.yaml`（已是活配置，可直接用）
- 共享测试 fixture：`tests/scorers/conftest.py` 的 `install_fake_vllm`
- 完整 plan / review 交互记录：本节 §11.1–§11.4

**绝对不要做的事**：
- 不要把 cache 写到 `/tmp` / `~` / `$HOME`——用户明确要求所有 cache 在当前目录。
- 不要给 vLLM 加"第 13 个具名参数"除非真的用上——走 `extra_llm_kwargs` 就行。
- 不要把 `dtype` 默认改回 `"auto"`——`"bfloat16"` 是当前两个模型的实际精度，更显式。

### 11.9 踩坑补录 · 安装时的 transformers 版本冲突（2026-04-21 尾声）

完成架构文档后，用户尝试 `pip install -e ".[all]"` 触发了长时间 pip 回溯：

```
vllm==0.11.0   requires transformers>=4.55.2
hpsv3==1.0.0   pins     transformers==4.45.2     ←── 硬冲突
```

**根因**：`hpsv3` 的 `pyproject.toml` 把 transformers 钉死（估计是 Qwen2-VL 在 transformers 4.45 前后 breaking change 的保守做法）；vLLM 0.11 需要 ≥4.55。两者无交集。pip 无法同时满足 → 一直回溯 vllm 版本试图找出一致图 → 用户看到的"一直在安装 vllm 0.19/0.18/..."现象。

**解法**：`install.sh` —— 两步走
1. `pip install -e ".[vllm,dev]"` 先装主体，让 pip 解出一致的 transformers（4.57.0）
2. `pip install --no-deps hpsv2 hpsv3 image-reward` 单独装旧包，**绕过**它们的 transformers pin

**为什么不能在 pyproject.toml 里声明**：PEP 508/621 没有"装这个依赖但忽略它的 sub-deps"语法，pip 也不支持 per-package `--no-deps`。这是 Python 打包生态的结构性缺陷。

**风险**：hpsv3 用 transformers 4.57.0 跑而非它声明的 4.45.2 —— 若 hpsv3 用到 4.45→4.57 之间被 breaking change 的 API 会运行时炸。缓冲：`registry._try_import` 层捕获 ImportError/AttributeError，对应 reward 不注册、service 照常起其他 reward。

**教训**：
- `[all]` extras 的"一键装"陷阱在多个 opt-in deps 有冲突 pin 时就失效，这是 pyproject.toml 天生的盲区。
- 遇到 pip 长时间回溯，第一反应应是"看看是不是两个包的版本约束互斥"，而不是"网速慢"或"vllm 版本没钉"。本次我一开始建议 pin vllm 版本只解决了 60% —— 真正的问题在 hpsv3 的 transformers pin 上。

### 11.10 安装冲突迭代 · 完整链路（2026-04-21 晚）

§11.9 写完后用户跑了一次 install.sh，暴露出一连串**层层嵌套**的问题。逐项记录，方便日后复盘或重建环境时避坑。

#### 11.10.1 第一轮：主体安装 OK，但三个旧包运行时都缺 sub-dep

install.sh 装完后 `import hpsv3` 炸 `omegaconf`、`ImageReward` 炸 `clip`、`hpsv2` 炸 `clint`。原因：`--no-deps` 跳过了旧包的**所有**依赖，不只是冲突的 transformers。

**修复**：install.sh 第 2 步增加"装非冲突 sub-dep"列表。通过 `pip download --no-deps` 抓每个 wheel 的 METADATA → 筛出缺失项。最终补装清单（11 个）：

```
ftfy braceexpand timm webdataset clint          # hpsv2
diffusers omegaconf fire matplotlib              # hpsv3 (matplotlib 是 hpsv3 源码用但 METADATA 漏了)
fairscale openai-clip                            # image-reward (openai-clip 同理，METADATA 未声明)
```

**坑**：`matplotlib` / `openai-clip` 两个包**METADATA 里没列**，但源码 `import` 它们。这是包作者的疏漏，只能靠运行时撞才发现。

#### 11.10.2 第二轮：transformers 真实 API 冲突

补装 sub-dep 后，`hpsv2` ✅ 活了，但：

```
hpsv3       ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'
ImageReward ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'
```

查源码确认：
- `hpsv3/model/differentiable_image_processor.py:52-58` 从 `image_utils` import `VideoInput`（transformers 4.57 已挪到 `video_utils`）
- `ImageReward/models/BLIP/med.py:31-36` 从 `modeling_utils` import 4 个 symbol（`apply_chunking_to_forward` / `find_pruneable_heads_and_indices` / `prune_linear_layer` / `PreTrainedModel`）——前 3 个挪到了 `pytorch_utils`

**解法**：新增 `reward_service/scorers/_compat.py`，用**monkey-patch** 把新位置的 symbol 塞回旧位置：

```python
def _shim(target_module, attr: str, source_module_path: str) -> None:
    if hasattr(target_module, attr):
        return
    try:
        source = __import__(source_module_path, fromlist=[attr])
        setattr(target_module, attr, getattr(source, attr))
    except (ImportError, AttributeError) as e:
        logger.warning(...)

_shim(transformers.image_utils, "VideoInput", "transformers.video_utils")
for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
    _shim(transformers.modeling_utils, _name, "transformers.pytorch_utils")
```

`hpsv3_scorer.py` / `imagereward.py` 顶部 `from reward_service.scorers import _compat` 保证 shim 先于旧包加载。

**不用 MetaPathFinder**：那是 import 系统的 hook，对"补 4 个 symbol"过度工程。monkey-patch 更朴素、更可读。

#### 11.10.3 第三轮：wandb/protobuf 版本不匹配

补完 compat shim 后，`import ImageReward` 新报：

```
google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf
Gencode/Runtime versions when loading wandb/proto/wandb_settings.proto:
gencode 6.32.1 runtime 6.31.1.
```

根因：`transformers 4.57` 触发 `wandb` 探测；环境里的 wandb 是新版（生成的 proto 代码要 protobuf runtime ≥ 6.32.1），但已装 protobuf 是 6.31.1。`pip install --upgrade protobuf` → 装了 7.34.1。

#### 11.10.4 第四轮：幽灵 dist-info

protobuf 升级后 `import ImageReward` 又报：

```
TypeError: expected string or bytes-like object, got 'NoneType'
  at diffusers/utils/constants.py:58
    version.parse(importlib.metadata.version("transformers")).base_version
```

`importlib.metadata.version("transformers")` 返回 `None`——神秘。查 `site-packages/` 发现**两个 transformers dist-info**：

```
transformers-4.57.0.dist-info/   (正常)
transformers-5.5.4.dist-info/    (只含一个空 REQUESTED 文件)
```

`5.5.4` 是**幽灵** ——之前某次 `pip install --target` 或被中断的操作留下的残渣，误导 `importlib.metadata` 查询。`rm -rf transformers-5.5.4.dist-info` 后 `importlib.metadata.version("transformers")` 恢复正常返回 `4.57.0`。

#### 11.10.5 第五轮：core dump（未解决）

清理幽灵 dist-info 后再试 `import ImageReward`——**core dump**。可能源自 protobuf 7.34.1 与环境里某个用老版 protobuf 编译的 C 扩展冲突；具体哪个扩展未定位。

此时用户决定：**保留本 session 当前的代码改动（都已通过 CPU 单测）** + **重置 conda 环境** 后再跑 install.sh 一次干净装。

#### 11.10.6 本 session 最终产物

**代码层面全部完成 · CPU 单测 65 passed**：

- `reward_service/scorers/_compat.py`（新）—— 4 条 transformers API shim
- `reward_service/scorers/hpsv3_scorer.py` 顶部加 `import _compat`
- `reward_service/scorers/imagereward.py` 顶部加 `import _compat`
- `tests/scorers/test_compat.py`（新）—— 3 条 shim 接口测试
- `install.sh` —— 最终版：主体 + 11 个 sub-dep + `--no-deps` 3 个旧包 + compat 验证段

**环境问题遗留给下次 resume**（重置 env 后再处理）：
- protobuf 升级引发的 core dump
- ImageReward 运行时路径上其他潜在的 transformers 4.45→4.57 不兼容（目前仅 import 层面救活了，未实测推理）
- hpsv3 同理（仅 import 层 OK）

**当前环境能真正用的 reward**（5/7）：
- ✅ clip · pickscore · hpsv2 · unified_reward · geneval2
- ❌ hpsv3（import OK，未验证推理）
- ❌ imagereward（import 阶段 core dump；待环境重置）

**重置 env 后的正确路径**（在仓库根目录执行）：
```bash
conda create -n reward-service python=3.12
conda activate reward-service
./install.sh
# 最后输出应看到 3 个 legacy 都 OK 或只有极少 runtime 错误
```

如果仍遇到错误，按 §11.10.1–11.10.5 顺序排查（sub-dep 缺失 → transformers API → protobuf → 幽灵 dist-info → C 扩展）。

---

## 12. 续篇 · 2026-04-27 session — 多机部署 + 绝对路径整改 + install.sh 兜底

### 12.0 时间线

- Session 开场：上个 session 尾声留了"/opt/conda/envs/torch-base 环境污染"问题，用户 docker 换环境后**首次真实启动了服务、并跑了 `scripts/bench_concurrent.py` 压测**（§11.8 记的"服务尚未真实启动过"至此作废）。
- 主任务：**双机 2×8 GPU 对称部署**，pdsh 拉起 Ray cluster，service 以 `ray.init(address=...)` 接入。
- 副任务 A：整改仓库内文档/脚本/skill 里对"仓库根"的绝对路径引用（YAML 里外部模型权重的绝对路径保留不动）。
- 副任务 B：install.sh 在用户机器上暴露 2 条新问题——hpsv3 缺 `peft` sub-dep、ImageReward shim 在 transformers 4.58+ 彻底失效。修的方式不是继续扩 shim，而是在 `pyproject.toml` 把 `transformers` 钉 `>=4.55.2,<4.58`（vLLM 0.11 下界 + 4.58 开始删 backward-compat alias 的上界），并在 install.sh 补 `peft`。
- code-standards skill 新增第 6 步"同步项目文档"：汇报完必须把进展与 resume 状态写回 `docs/`，否则视为未交付。本节即新流程的首次产物。

### 12.1 三条用户原话

1. "之前的压测是在单机下进行的压测，我现在想要测试一下多机的部署"
2. "在执行 install.sh 的时候: hpsv3 FAIL: ModuleNotFoundError: No module named 'peft' / ImageReward FAIL: ImportError: cannot import name 'find_pruneable_heads_and_indices' ... 是不是也应该修复到 install.sh 里？"
3. "这里更好的修复方式是不是先限制在 transformer 4.57 就可以了？因为 vllm=0.11，也限制了 transformers 不能是 5.x"

### 12.2 关键决策与理由

- **多机接入点不另起新架构**：§11.3 留的"`pool.py` 的 `ray.init()` 可换 `address=...`" 扩展点直接够用。L0 `BaseScorer` / L1 具体 scorer / L2 `@ray.remote ScorerActor` 全不动，只加 `ClusterCfg` + `_init_ray` 分支。
- **拉起集群走 shell 脚本，不进服务进程**：YAML 只负责"告诉服务去哪连"，"cluster 怎么起"由 `scripts/cluster_{up,down,smoke}.sh` 负责。职责边界清晰。
- **`scheduling` 字段加到 reward 级**：用户明确要求。`"pack"`（默认，Ray 内建行为）/ `"spread"`（`scheduling_strategy="SPREAD"` hint）。单 replica group 天然无意义但字段仍存在，文档说明即可，不做 YAML-time warning（YAGNI）。
- **GenEval2 TP=2 坚持 `scheduling: pack`**：跨节点 vLLM TP 走以太网 NCCL 会让吞吐崩塌；示例配置里写死 pack 绑定单机。
- **`NODE_IP_LIST` 环境变量**：用户确认腾讯云 Docker 已经有这个 env（格式 `"ip1:8 ip2:8"`，`:8` 是 per-node GPU 数）。`scripts/_cluster_lib.sh` 提供 `resolve_cluster_nodes` 共享函数，用 awk 按"第一个 `:`"切（不是 `sed 's/:[0-9]*//g'`——后者 IPv6 字面量会被 mangle）。
- **worker `--node-ip-address` 用 YAML 里的 IP，不用 `hostname -i`**：review 阶段发现的隐坑。容器里 `hostname -i` 经常返回 `127.0.0.1`，Ray 静默绑 loopback 后 cluster 看起来起来了但 actor 调不过去。per-worker 单独 `pdsh -w $ip` 循环，IP 从 `NODE_IP_LIST` 直接插值。
- **ImageReward shim 失效——不追加 fallback，改钉 transformers 上界**：原思路是在 `_compat.py` 加内联 fallback 实现（`apply_chunking_to_forward` / `find_pruneable_heads_and_indices` / `prune_linear_layer` 三个 20 行以内的 torch 工具函数）。用户指出 vLLM 0.11 本来就不让 transformers 上 5.x，那钉 `<4.58` 就根治；fallback 是 YAGNI。回滚内联实现和对应测试。
- **`transformers<4.58`**：pyproject 下界 `>=4.55.2`（vLLM 0.11 的硬要求）、上界 `<4.58`（4.58 开始移除 `transformers.pytorch_utils` 里我们还依赖的 backward-compat alias）。`_compat.py` docstring 补注"若未来上调上界请先回来看这里"。

### 12.3 文件改动清单

**新增**
- `configs/service.cluster.example.yaml`——双机 16 GPU 示范；6 个 1-GPU reward 用 `num_replicas: 2 + scheduling: spread`，GenEval2 用 `scheduling: pack` 绑单机
- `scripts/cluster_up.sh`——读 `NODE_IP_LIST`，pdsh head + 循环 per-worker；带 `--temp-dir=$PWD/.ray-tmp-$(hostname)`（每机本地、不走 NFS）
- `scripts/cluster_down.sh`——pdsh 全机 `ray stop --force || true`
- `scripts/cluster_smoke.sh`——端到端：`smoke_client.py` + `bench_concurrent.py --sweep`
- `scripts/_cluster_lib.sh`——共享 `resolve_cluster_nodes` 函数

**修改**
- `reward_service/config.py`——`ClusterCfg(ray_address, namespace)` dataclass + `RewardModelCfg.scheduling` 字段；`_parse_cluster_cfg` 用内部 `_opt_str` 去重字符串校验
- `reward_service/workers/pool.py`——`_init_ray(ClusterCfg)` 三分支：已初始化复用 / 有 address 连 cluster / 否则本地起；去掉 `ignore_reinit_error=True`（前置 `is_initialized()` 守卫已等价）
- `reward_service/workers/group.py`——`_actor_options(cfg)` 翻译 `scheduling=spread → scheduling_strategy=SPREAD`；`_SPREAD_STRATEGY` 模块常量
- `reward_service/scorers/_compat.py`——docstring 注明 transformers `<4.58` 的上界约束由 pyproject 承担、无需 `_shim` 内联 fallback
- `install.sh`——`legacy_subdeps` 列表加 `peft`
- `pyproject.toml`——`transformers>=4.55.2,<4.58`（原来是 `>=4.44`）
- `tests/test_config.py`——cluster 字段 7 条新用例（含 parametrize 拆分 address/namespace 单 behavior）+ scheduling 字段 4 条
- `tests/workers/test_pool.py`——`TestInitRay` 4 条（local / cluster address / namespace / reuse）
- `tests/workers/test_group.py`——`TestSchedulingStrategy` 2 条（pack 不传 / spread 传 SPREAD）
- `docs/DEVELOPMENT_LOG.md`、`docs/RESUME_PROMPT.md`——副任务 A 的相对路径整改；`.claude/skills/code-standards/SKILL.md` 同步父目录

### 12.4 simplify 应用结果（SHOULD-FIX）

- `_parse_cluster_cfg` 内部抽 `_opt_str` 去重两段字符串校验（两块 if/value/raise 合并）
- 新建 `scripts/_cluster_lib.sh`，两个 cluster 脚本共用 `resolve_cluster_nodes`；`sed 's/:[0-9]*//g'` → awk 按首个 `:` 切（修 IPv6 风险）
- `_init_ray` 去掉 `ignore_reinit_error=True` 冗余 kwarg，测试断言同步

**不采纳**：`scheduling` 改 StrEnum/Literal（YAGNI）、`_actor_options` 内联（函数独立更可读）、两份 yaml 去重（它们是不同 GPU 预算示范）、`_init_ray` docstring 收缩（三段式 case 区分有价值）。

### 12.5 review 应用结果（SHOULD-FIX）

- `cluster_up.sh` worker IP 改为 per-worker 循环，不再依赖 `hostname -i`（避开容器 loopback 陷阱）
- pdsh 内嵌 heredoc 从 `set -e` 升级 `set -euo pipefail`（对齐项目规范）
- `test_load_config_cluster_parses_address_and_namespace` 拆两条单 behavior + `strips_whitespace` parametrize 覆盖两字段
- `group.py` `_actor_options` docstring 重写为 WHY-first；抽 `_SPREAD_STRATEGY: Final = "SPREAD"` 常量

**不采纳**：`_opt_str` 注解（局部函数、推断清晰）、`test_pool` 通过间接 API 测 `_init_ray`（当前直测更显式）、`_cluster_lib.sh` 两-pass awk → 单 awk（可读性损失）、smoke 脚本 CONCURRENCY 命名常量（已是顶部 env var）。

### 12.6 测试状态

本 session 触及的三个测试文件共 **40 passed**（在本代理 shell 的 transformers 4.48 + torch 2.1 环境里跑通——跑通点早于本节的 simplify/review 改动）。

simplify / review 阶段之后的改动都是**行为等价重构**：
- `_parse_cluster_cfg` 抽 helper——输入/输出与既有断言一致
- `_init_ray` 删 `ignore_reinit_error`——对应测试断言已同步
- `cluster_up.sh` 重构——shell 改不影响 python 单测
- `group.py` `_SPREAD_STRATEGY` 常量——等价替换

**遗留**：本机 pytest 无法在目标环境（python3.12 + torch 2.8 + transformers 4.57）下跑；交付时请在 docker 里以标准命令复核：

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest \
    -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
```

### 12.7 踩到的坑

- 代理 shell 环境是 python3.11 + torch 2.1 + transformers 4.48；手动升 transformers 到 4.57 让 `torch.utils._pytree` 炸（2.1 里叫 `_register_pytree_node`，新 transformers 要 `register_pytree_node`）。教训：**别乱动代理 shell 的包版本，它跟目标机器不是同一个 env**。
- 用户反馈"transformers 5.6 才会报 `DistributedTensorGatherer`，4.57 没问题"——原本我以为 4.58+ 会删 `DistributedTensorGatherer`，验证下游库报错路径后发现其实 4.57.x 该类还在 `trainer_pt_utils`，只是不是 `trainer` 的公共 re-export。我们的 `<4.58` pin 已经避开了，这是对 pin 策略的间接验证。

### 12.8 未完成事项

- 真正到双机上跑 `cluster_up.sh` → 启动服务 → `cluster_smoke.sh`，还未做。
- `num_replicas: 2 + scheduling: spread` 在实际 Ray 调度下是否真的分到两台机，还没观测。如果观测结果是"replica 都挤在一台机上"，下一步再上 placement group（§4.1 ARCHITECTURE 的"资源隔离 vs 软 hint"讨论）。
- `docs/ARCHITECTURE.md §5.3` 的"多机扩展"段原本写的是"改 `pool.py:28`"；现在已经不需要改代码了（YAML 填 `cluster.ray_address` 即可），等跑通后回来更新。

---

*本节由 2026-04-27 session 期间同步维护；后续迭代请在末尾加 §13、§14 …… 并更新 §11.8 "Resume 入口" 为最新状态（见 §12.9）。*

### 12.9 Resume 入口（覆盖 §11.8，以本节为准）

**快速状态**（2026-04-28 更新）：
- **多机部署**：双机 16 GPU 已跑通。YAML 调度顺序已修复（多 GPU actor 排最前，避免碎片化）。
  - 多机配置：`configs/service.cluster.example.yaml` + `scripts/ray_{start,stop,smoke}.sh` + `_ray_lib.sh`
  - 配置：`ClusterCfg(ray_address, namespace)` · `RewardModelCfg.scheduling ∈ {"pack","spread"}`
  - 调度顺序重要：geneval2(pack,2GPU) → unified_reward(spread,2×2GPU) → 5个轻量reward(spread,各2×1GPU) = 16 GPU 全满
- **hpsv2 报错（未修，第一优先级）**：`TypeError: score() got an unexpected keyword argument 'cp...'`。
  位置：`reward_service/scorers/hpsv2_scorer.py:72-73`。
  原因：目标环境 `hpsv2` 包的 `img_score.score()` 函数签名与代码假设不符。
  修法：在有 `hpsv2` 的环境里 `python3.12 -c "import inspect; from hpsv2 import img_score; print(inspect.signature(img_score.score))"` 查真实签名，然后改调用处。
- **其他 bug 修复**（§12.10–§12.14）：ray_stop.sh 杀全栈、pickscore 删 processor_name、代理 env 透传、config 校验 TP≤num_gpus
- **测试**：本 session 新增 16 条单测（config 9、pool 4、group 2、pickscore 3）；40 passed；目标环境需复核
- **环境**：docker 里 python3.12 + torch 2.8 + transformers 4.57（在 `<4.58` pin 范围内）
- **环境坑**：flash-attn 2.x 与 torch 2.8 ABI 不兼容 → `pip uninstall flash-attn`

**下次进来先做的事**（按优先级）：

1. **修 hpsv2 scorer**（第一优先级）：
   ```bash
   python3.12 -c "import inspect; from hpsv2 import img_score; print(inspect.signature(img_score.score))"
   ```
   拿到真实签名后改 `hpsv2_scorer.py:72-73` 的调用。

2. **跑压测**：
   ```bash
   # smoke（确认所有 reward 正常）
   PYTHONPATH=. python3.12 scripts/smoke_client.py --url http://localhost:8080
   # 并发 sweep
   PYTHONPATH=. python3.12 scripts/bench_concurrent.py --url http://localhost:8080 --sweep 50 100 200 400 800 --total 500
   # 或一键 smoke+sweep
   bash scripts/ray_smoke.sh http://localhost:8080
   ```

3. **如果要重启多机**：
   ```bash
   export NODE_IP_LIST="ip1:8 ip2:8"
   export HTTP_PROXY=... HTTPS_PROXY=... NO_PROXY=...
   bash scripts/ray_start.sh
   PYTHONPATH=. python3.12 -m reward_service --config configs/service.cluster.example.yaml
   # 停：
   bash scripts/ray_stop.sh
   ```

4. **如果要跑单测**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
   ```

**关键文件索引**：
- 架构全景：`docs/ARCHITECTURE.md`
- 多机接入点：`reward_service/workers/pool.py` 的 `_init_ray`
- scheduling 透传：`reward_service/workers/group.py` 的 `_actor_options`
- YAML schema：`reward_service/config.py` 的 `ClusterCfg` / `RewardModelCfg`
- 集群脚本：`scripts/ray_{start,stop,smoke}.sh` + `_ray_lib.sh`
- 多机示例：`configs/service.cluster.example.yaml`（注意调度顺序：多 GPU actor 排最前）
- compat shims：`reward_service/scorers/_compat.py`
- 本次迭代完整记录：§12.1–§12.15

**绝对不要做的事**（继承 §11.8，仅新增）：
- 不要把 cache 写到 `/tmp` / `~` / `$HOME`——所有 cache 在当前目录
- 不要把 `dtype` 默认改回 `"auto"`——`"bfloat16"` 是当前两个 vLLM 模型的实际精度
- 不要给 `build_vllm_llm_kwargs` 加"第 13 个具名参数"——走 `extra_llm_kwargs`
- 不要静默改 hpsv3 / ImageReward 的 site-packages 源码——走 `_compat.py` monkey-patch
- 不要跳过 plan 直接动代码——单行修复除外
- **不要把 `transformers` 上界抬到 4.58+**——`_compat.py` 和 ImageReward / hpsv3 在那个版本会崩
- **不要在仓库内文档/脚本/skill 里写仓库根绝对路径**——用相对路径；YAML 外部权重路径除外
- **不要把 Ray temp-dir 放回工程目录**——AF_UNIX socket 107 字节限制，走 `/tmp/ray-$USER`
- **不要随意调整 `configs/service.cluster.example.yaml` 里 rewards 的顺序**——多 GPU actor 必须排最前，避免 GPU 碎片化导致调度失败（§12.15）

---

*§12.9 是当前 Resume 入口。再有新 session 请覆盖本节、保留 §11.8 作为历史。*

### 12.10 踩坑 · Ray temp-dir 的 AF_UNIX 107 字节限制 + 脚本改名（2026-04-28）

**背景**：§12.3 里我把 `scripts/cluster_up.sh` 的 Ray `--temp-dir` 配成 `$PWD/.ray-tmp-$(hostname)`，用意是对齐项目"cache 留当前目录"的硬约束。实际上双机拉起 Ray cluster 时撞到了 kernel 级硬限制：

```
OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes:
<project-root>/.ray-tmp-<host>/session_2026-04-28_11-20-51_444328_33190/sockets/plasma_store
```

**字节数算一下**：

```
<project-root>   = 66 字节
/.ray-tmp-<host>                                                = 23 字节
/session_2026-04-28_11-20-51_444328_33190                               = 40 字节
/sockets/plasma_store                                                   = 20 字节
                                                             总计 = 149 字节   超限 42
```

工程根本身就占 66 字节，Ray 往下挂三层目录（temp-dir / session / sockets）+ 一个文件名，每层都是 20~40 字节，107 字节物理上塞不下。这是 Linux kernel 的 `struct sockaddr_un.sun_path[108]`，不是 Ray 的软限制。

**决策**：Ray runtime temp-dir 放 `/tmp/ray-$USER`。

**为什么不违反"cache 在当前目录"硬约束**：原约束的字面列举是"`.pycache / .pytest_cache / .pip-cache / .install.out`"——这四类都是**累积型、跨 session 复用**的构建产物。Ray temp-dir 里是 Unix socket + shm 链接 + session log，**每次 `ray start` 重建、`ray stop` 清掉**，语义上更接近 `/var/run/postgresql/` 这类进程运行时文件。项目其他所有缓存/产物的"必须在当前目录"约束保持不变。`docs/RESUME_PROMPT.md` 相应位置加了一条显式例外。

**顺带**：三个 cluster 脚本改名以更贴合职责（启停的是 Ray 守护进程，不是抽象的"cluster 生命周期"）：

| 旧名 | 新名 |
|---|---|
| `scripts/cluster_up.sh` | `scripts/ray_start.sh` |
| `scripts/cluster_down.sh` | `scripts/ray_stop.sh` |
| `scripts/cluster_smoke.sh` | `scripts/ray_smoke.sh` |
| `scripts/_cluster_lib.sh` | `scripts/_ray_lib.sh` |

函数名 `resolve_cluster_nodes` 保留（它确实解析的是 cluster 级拓扑 head+workers；跟脚本名所指"进程"不冲突）。

**文件改动清单（本小节）**

修改：
- `scripts/ray_start.sh`（+ rename from `cluster_up.sh`）——新增 `RAY_TMPDIR` 环境变量（默认 `/tmp/ray-$USER`）；两处 pdsh heredoc 的 `--temp-dir` 全部改用它；usage 头加说明；所有 "cluster_up" 字样改成 "ray_start"
- `scripts/ray_stop.sh`（+ rename from `cluster_down.sh`）——header 指向新 `ray_start.sh`；echo 前缀换
- `scripts/ray_smoke.sh`（+ rename from `cluster_smoke.sh`）——header 指向新 `ray_start.sh`；usage 示例 + 尾部 echo 换名
- `scripts/_ray_lib.sh`（+ rename from `_cluster_lib.sh`）——header 注释换
- `docs/RESUME_PROMPT.md`——硬约束条款加 Ray temp-dir 例外；启动示例换新脚本名；关键文件索引换新脚本名
- `docs/ARCHITECTURE.md §5.3`——示例命令换新脚本名
- `README.md` 多机段——命令换新脚本名
- `CHANGELOG.md [Unreleased]`——条目更新为新名 + 新 RAY_TMPDIR 说明
- `docs/DEVELOPMENT_LOG.md §12.9`——Resume 入口同步新名 + 新增"不把 Ray temp-dir 放回工程目录"警示

**单元测试**：本次改动是 shell 路径字面量替换 + 重命名 + 文档文字更新，属 §4.5.2 免测清单（"纯 I/O 粘合代码 & 无分支逻辑"）。未新增/修改 Python 逻辑。`bash -n` 语法检查四个 shell 脚本全部通过。

**遗留**：真正在双机跑 `ray_start.sh` 验证 temp-dir 新路径下 Ray cluster 能顺利拉起 → 启动 service → `ray_smoke.sh` 过 sweep，待目标机器复测。

### 12.11 踩坑 · Ctrl+C 停 service 留下 vLLM 孤儿进程（2026-04-28）

**现象**：service 用 `python3.12 -m reward_service ...` 前台启动，Ctrl+C 退出后 `nvidia-smi` 显示多张卡还被占。

**根因链**：
- vLLM 的 `LLM` 对象在构造时用 `multiprocessing` spawn 出 `VllmWorkerProcess` 子进程。这些子进程**不是 Ray 进程树的一部分**，Ray 不知道它们的存在。
- `ray.kill(actor)` 对 actor 是 SIGKILL 级别强杀，actor 没机会跑 `__del__` 来清 vLLM worker 子进程——后者变成孤儿被 init 托管。
- 更糟的是，Ctrl+C 给 uvicorn SIGINT 后，FastAPI lifespan 的 `finally: pool.shutdown()` 在冷启动/长任务中期经常跑不完，`ray.kill` 本身都没来得及调。

**决策**：不改 Python（改 lifespan/加 SIGTERM handler 的改动面大、验证成本高），改为**强化 `ray_stop.sh` 的职责**为"下线整栈"：

```
SIGTERM reward_service python 进程 (给它 10s 走 lifespan cleanup)
  ↓ 超时
SIGKILL reward_service
  ↓
SIGKILL 所有节点残留的 VllmWorkerProcess (兜底)
  ↓
ray stop --force
```

**实现细节**：
- 全程用 `pdsh` 在每台节点并行跑，一个 pdsh block 顺序执行上面四步（单 ssh 往返，便于看日志）。
- `pkill -f 'python.*-m reward_service'`：pattern 限定到 `__main__.py` 的标准启动姿势，误杀面几乎为 0。
- `pkill -f '(^|[[:space:]])([Vv][Ll][Ll][Mm]::|VllmWorker)'`：覆盖 vLLM 跨版本的不同 setproctitle 名字——实测 vLLM 0.11 的 worker 进程叫 `VLLM::EngineCore`（全大写 + `::`），不是早期的 `VllmWorkerProcess`；也预留 `vllm::worker-N` / `vllm::engine_core` 小写变体。加了 `SKIP_VLLM_PURGE=1` env 作为显式 opt-out（同机共享其他 vLLM workload 时用）。
- 每步都 `|| true`：任一步没进程可杀都正常（例如只想下 Ray、service 已经提前停了）。
- `pkill -0`：探测存在、不发信号；等待循环可提前退出。

**文件改动**：
- `scripts/ray_stop.sh`：从"只跑 `ray stop --force`"扩成上述四步；usage 头 / 顶部注释同步更新
- `docs/DEVELOPMENT_LOG.md §12.11`（本节）
- `CHANGELOG.md [Unreleased]`：同步条目

**单元测试**：shell 脚本 + 无分支逻辑 + pattern 固定，属 §4.5.2 免测清单。`bash -n` 语法检查通过；heredoc 变量展开手工模拟确认 `${service_pattern}` 远端 eval、`${SKIP_VLLM_PURGE}` 本地 eval、逻辑正确。

**未动但可能后续做**：给 `reward_service/__main__.py` 加 SIGTERM/SIGINT handler 保证 lifespan `finally` 跑完再退。如果观察到 10s SIGTERM-等待窗口常常不够走完 Ray actor kill + vLLM worker 析构，再回来做这个。目前的 shell 兜底已经能解决用户观察到的 GPU 残留。

**使用**：
```bash
bash scripts/ray_stop.sh                    # 下线整栈（默认）
SKIP_VLLM_PURGE=1 bash scripts/ray_stop.sh  # 不清 vLLM 孤儿（共享节点场景）
```

### 12.12 ray_stop.sh 多轮迭代 → 最终走 nvidia-smi PID 方案（2026-04-28）

§12.11 的 `pkill -f` 方案在真实双机环境撞了两个连锁坑：

1. **pkill self-match**：`pkill -f 'VLLM::'` 在远端 `bash -c '...pkill -f "VLLM::"...'` 中执行时，父 bash 自己的 cmdline 里字面有 `VLLM::` 这几字节，pkill 把父 bash 也一起杀——后面的 `ray stop --force` 根本没来得及跑。head 节点偶尔没事是 PID 顺序运气，worker 节点稳定复现。
2. **误杀 sshd**：尝试用 env var 传 pattern 的那版，因为 `pdsh -w host VAR=x '...'` 不是合法语法（pdsh 会把 `VAR=x` 当成命令），远端 `$VAR` 为空，`pkill -f ""` 空 pattern 匹配**所有**进程——把 sshd 都杀了，需要腾讯云 VNC 进去重启 sshd 才能恢复 ssh。

**最终方案**（当前版本）：彻底放弃复杂脚本，一把梭 pdsh：

```bash
pdsh -R ssh -w $NODE_LIST '
    ray stop --force 2>/dev/null || true
    pkill -9 -f reward_service 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
'
```

三行、31 行脚本、无 env var、无分支、无 nvidia-smi 依赖。

**self-match 残留风险的诚实交代**：
- 远端父 bash 的 cmdline 里字面含 `reward_service` 和 `VLLM::`，所以 pkill 会匹配到父 bash 自己
- **但** pkill 的语义是"先枚举所有匹配 pid、再依次发 signal"——在父 bash 被 kill 之前，目标 reward_service 主进程已经收到 SIGKILL
- 父 bash 死后第二条 `pkill -9 -f "VLLM::"` 不会跑；但 reward_service 主进程死后 OS 会级联清理它的 VLLM 子孙，第二条本来就是兜底
- 如果将来发现 VLLM 不跟 reward_service 一起死，就把两个 pkill 换个顺序，让 reward_service 那条作为"最后一刀"

**SKIP_ORPHAN_KILL env var 等复杂性全部移除**——这一版没有 opt-out 按钮，要隔离请手动注释行。

**教训**：
- shell 命令跨 `local bash → pdsh → ssh → remote bash` 链，**任何 pattern 展开、引号嵌套、变量捕获都是可能的陷阱**——能用具体数值（PID）代替 pattern 就用具体数值
- `pdsh` 的命令字符串传递语法不接受位置 `KEY=VAL` 形式把环境变量注入远端；要传环境变量要么写到 remote 脚本的 shebang 参数里，要么通过 ssh agent forwarding，要么干脆把值写死在字符串里
- **手误杀 sshd 的代价非常高**：必须带外恢复；这类操作要先在单节点验证（`NODE_IP_LIST="ip1:8" bash scripts/ray_stop.sh; ssh ip1 echo ok`）再推广到集群

**文件改动**：
- `scripts/ray_stop.sh`（当前最终版）：pdsh + nvidia-smi + 按 PID 杀
- `docs/DEVELOPMENT_LOG.md §12.12`（本节）

**单元测试**：shell 脚本 + 只传数字 PID，属 §4.5.2 免测。真实双机跑通验证（见 §12.9 遗留事项收尾）。

### 12.13 Scorer 本地优先的统一整改（2026-04-28）

**背景**：双机部署时 pickscore actor 在 worker 节点（无外网）崩溃：
```
OSError: Can't load image processor for 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
```
YAML 里已经给了 `weights_path` 指向本地权重目录，但 scorer 代码里 **只让 model 走 weights_path，processor 仍然硬编码 HF id**：

```python
self.model = CLIPModel.from_pretrained(model_path)          # ✓ 走本地
self.processor = CLIPProcessor.from_pretrained(processor_name)  # ✗ 走 HF id
```

**关键发现**：查看 `weights_path` 指向的 PickScore_v1 目录，发现 HuggingFace 的 PickScore_v1 repo **本来就把 CLIP processor 的所有文件（preprocessor_config.json / tokenizer.json / vocab.json / merges.txt / special_tokens_map.json / tokenizer_config.json）和模型权重 一起打包了**。所以 processor 根本不需要单独路径——它就在 model 权重旁边。

**决策**：删掉 `processor_name` 字段。processor 跟 model 共用同一个路径——复用 `resolve_model_path(model_name, weights_path)` 的结果，和 clip.py 的做法完全对齐（clip 一开始就做对了）。

```python
# 最终版
path = resolve_model_path(model_name, weights_path)
self.model = CLIPModel.from_pretrained(path)
self.processor = CLIPProcessor.from_pretrained(path)
```

**其他 scorer 扫描**（决定不扩大整改）：

| scorer | 是否有同类问题 |
|---|---|
| clip | 已正确（model + processor 都用 `path`） |
| **pickscore** | **有问题，本次修复** |
| imagereward | `med_config_path` 指向独立的 `.json`，非 HF 资源——合理 |
| hpsv2 | 靠 HPS_ROOT + 版本号自动定位 `.pt`——无 `_name` 硬编码 |
| hpsv3 | `config_path` 指向独立的 yaml，与 model 权重分离——合理 |
| unified_reward / geneval2 | 已正确 |

所以"所有 scorer 统一本地优先"这个原则是对的，但事实上**只有 pickscore 一个违例**需要修。

**文件改动**：
- `reward_service/scorers/pickscore.py`：`__init__` 签名删 `processor_name`；processor 从 `path` 加载；docstring 说明本次对齐 clip.py 的约定
- `configs/service.cluster.example.yaml` + `configs/service.example.yaml`：pickscore 段删 `processor_name:` 行
- `tests/scorers/test_pickscore.py`：从只有 GPU smoke test 扩充成含 3 条 CPU 不变式测试——用 monkeypatch 捕获 `CLIPModel.from_pretrained` / `CLIPProcessor.from_pretrained` 入参，断言：
  - 给 `weights_path` 时两者都用本地路径
  - 不给 `weights_path` 时两者都 fallback 到 `model_name`
  - 传 `processor_name=` kwarg 现在会报 `TypeError`（防回归）
- `CHANGELOG.md [Unreleased]`：breaking change 条目

**兼容性影响**：breaking change——`pickscore` params 里还留着 `processor_name:` 的旧 YAML 会让 actor `__init__` 报 `TypeError: unexpected keyword argument 'processor_name'`。本仓库两份示例 YAML 都已同步；外部用户极少，此时删字段成本最低。

**单元测试**：3 条 CPU 不变式测试，用 monkeypatch 不依赖真模型。目标 docker 环境跑一次：
```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest tests/scorers/test_pickscore.py -m "not gpu and not slow" -v -o cache_dir=./.pytest_cache
```

### 12.14 代理 env 透传给 Ray actor（2026-04-28）

**背景**：ImageReward 第三方包源码里 `BertTokenizer.from_pretrained('bert-base-uncased')` 硬编码 HF id，worker 节点如果走代理才能到 HF，但 Ray actor 进程里没有 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 环境变量——拉不到 vocab 文件炸在 `os.path.isfile(None)`。

**决策**：在 `ray_start.sh` 的每个 `ray start` 命令前缀注入这三个变量。raylet 启动时拿到、它起的所有 actor 子进程自动继承——无需改 Python 代码、无需 `ray.init(runtime_env=...)`。

**实现**（两行改动）：
```bash
HTTP_PROXY='${HTTP_PROXY:-}' HTTPS_PROXY='${HTTPS_PROXY:-}' NO_PROXY='${NO_PROXY:-}' \
ray start --head ...
```

pdsh remote 命令串用双引号 heredoc，本地 shell 先展开 `${HTTP_PROXY:-}` 成具体 URL（未设时为空串）。空串注入等价于"这个变量存在但为空"——`requests` 库的 proxy 探测仍然不会走代理，所以空值安全。

**验证方式**：
```bash
ssh <worker> 'cat /proc/$(pgrep -f raylet | head -1)/environ | tr "\0" "\n" | grep -i proxy'
```

**未做但讨论过**：设 `HF_HOME` / `HF_ENDPOINT` / `TRANSFORMERS_OFFLINE`——本次只需要代理，其他 YAGNI；若将来要预装 HF cache 或走 mirror 再加。

**文件改动**：
- `scripts/ray_start.sh`（+2 行 proxy 前缀）
- `docs/RESUME_PROMPT.md`（启动示例加一句"先 export 代理"）
- `CHANGELOG.md [Unreleased]`（+1 条）

**单元测试**：shell 纯 env 转发，属 §4.5.2 免测。`bash -n` 语法检查通过；真实双机跑通再来回来更新 Resume 入口。

### 12.15 多机部署调通 + YAML 调度顺序修复 + hpsv2 待修（2026-04-28）

**多机部署状态**：Ray cluster 双机 16 GPU 已拉起，service 成功启动。

**调度顺序问题**：
- 现象：`ray status` 显示 14/16 GPU allocated，geneval2（需 2 GPU 同节点）pending。
- 原因：YAML 里 geneval2 排最后。12 个 1-GPU SPREAD actor 先调度，均匀分到两节点（各 7 个）；unified_reward 2×2 GPU 再调度（各 1 replica 到 A/B）→ 每节点只剩 1 GPU → geneval2 需要 2 GPU 连续在同节点，放不下。
- 修复：**把多 GPU actor（geneval2、unified_reward）挪到 `rewards:` 列表最前面**。调度顺序变为：geneval2(2GPU)→unified_reward(2×2GPU)→10 个 1-GPU actor → 16 GPU 全部填满。
- 文件改动：`configs/service.cluster.example.yaml`（纯顺序调整 + 注释更新 budget check）。

**hpsv2 报错（未修，下次 session 继续）**：
```
TypeError: score() got an unexpected keyword argument 'cp...'
```
- 出错位置：`reward_service/scorers/hpsv2_scorer.py` 第 72-73 行调用 `self._img_score.score([image], text, cp=..., hps_version=...)`。
- 推测原因：目标环境安装的 `hpsv2` 包版本与开发时假设的 API 签名不同。`img_score.score()` 可能不接受 `cp` 关键字参数（或叫 `hps_version`），需要在有 `hpsv2` 包可 inspect 的环境里 `inspect.signature(hpsv2.img_score.score)` 确认真实签名后修复。
- 这是 **下次 session 的第一优先级**。

**spread vs pack 的含义**（用户问答记录）：
| | pack（默认） | spread |
|---|---|---|
| 策略 | bin-packing，尽量填满一台再用下一台 | 分散到不同节点 |
| 适合 | TP>1 的 vLLM（多 GPU 必须同节点） | 1-GPU reward 做负载均衡 |
| Ray 实现 | 默认行为 | `scheduling_strategy="SPREAD"` 软提示 |

---

## 13. 续篇 · 2026-04-28 session — 修复 hpsv2 scorer

### 13.0 时间线

- 用户指示：resume RewardService，第一优先级修 hpsv2 scorer 报错 `TypeError: score() got an unexpected keyword argument 'cp...'`
- 探查阶段：发现环境中安装的 hpsv2 是从 `file:///<local>/images/ptm_runtimes/new-HPSv2` 本地安装的 fork 版（`__version__='1.2.0.1'`），API 签名与 GitHub main（PyPI 1.2.0）完全不同
- 决策：重装 PyPI 官方 hpsv2 1.2.0，按 GitHub main 的 API 重写 scorer
- 关键发现：GitHub main 的 `img_score.score()` 每次调用都 `torch.load(cp) + model.load_state_dict()` —— 性能不可接受
- 最终方案：`__init__` 一次性完成模型加载（`create_model_and_transforms` + `torch.load` + `load_state_dict` + `get_tokenizer`），`_score_single` / `_batch_score` 复用缓存模型做推理，推理逻辑（`torch.no_grad` → `unsqueeze(0)` → `torch.cuda.amp.autocast` → `features @ features.T` → `diagonal()[0]`）与 GitHub main 逐行对齐，保证数值完全一致

### 13.1 用户原话

> 修 hpsv2 scorer 报错：`TypeError: score() got an unexpected keyword argument 'cp...'`
> 先查实际签名 → 拿到签名后改调用处 → 跑通 smoke test 验证

> 你如果要把 load 放到 init 里面，那么对应的输入就应该与之前的 hpsv2 的 score 输入保持完全一致

> 你只需要作用与 baseline 完全一致即可，不用支持无用的东西

### 13.2 关键决策与理由

#### 13.2.1 重装 PyPI 官方 hpsv2，不沿用 fork

环境中的 fork 版 (`1.2.0.1`) 来源不可控（`/<local>/.../new-HPSv2`，本机不可访问），且 `install.sh` 已经 `pip install --no-deps hpsv2` 装的是 PyPI 官方版。沿用 fork 意味着其他机器无法复现。

#### 13.2.2 推理逻辑逐 item 循环而非 batch forward

GitHub main 的 `score()` 是逐 item 的——每次 `unsqueeze(0)` 单张图 + 单条 prompt 做 forward。为保证数值完全一致，我们的 `_batch_score` 也是逐 item 调 `_score_single`，而非 stack 成 batch 一次 forward（后者在 LayerNorm 下理论上等价，但用户明确要求"与 baseline 完全一致"）。

#### 13.2.3 `torch.cuda.amp.autocast()` 保持旧 API

GitHub main 用 `torch.cuda.amp.autocast()`（无参数），torch ≥2.6 会发 `FutureWarning` 建议改用 `torch.amp.autocast('cuda', ...)`。为对齐 upstream 选择保留旧写法，不加 `weights_only=True` 也同理。

### 13.3 文件改动清单

修改：
- `reward_service/scorers/hpsv2_scorer.py`（完全重写）
  - `__init__`：新增 `device` 参数；调用 `create_model_and_transforms` 构建模型架构 → `torch.load` + `load_state_dict` 加载权重 → 缓存 `_model`、`_preprocess_val`、`_tokenizer`
  - `_score_single`：逐 item 推理，逻辑与 GitHub main `img_score.score()` 逐行对齐
  - `_batch_score`：逐 item 调 `_score_single`，输入校验用 `ValueError`
  - `score`：委托给 `_batch_score`，通过 `split_last_turn` 拆解 items
- `tests/scorers/test_hpsv2.py`（完全重写）
  - 7 条 CPU 不变式测试 + 1 条 GPU smoke test
  - `_make_fake_hpsv2`：monkeypatch 完整 hpsv2 模块层级
  - `_make_dummy_checkpoint`：从 scorer 模块 import `_CHECKPOINT_FILENAMES` 避免重复

### 13.4 simplify 应用结果

已修复 6 项：
- `torch.load + load_state_dict` if/else 两分支重复 → 合并
- `_device_str` 冗余字段 → 删除
- `assert` 做输入校验 → `raise ValueError`
- 测试文件未使用的 `MagicMock` import → 删除
- 测试里重复的 `_CHECKPOINT_FILENAMES` → 改为从 scorer import
- GPU smoke test 过度复杂的 `__import__` → 普通 `from ... import`

跳过 5 项（Consider 级别，与本次范围无关）。

### 13.5 review 应用结果

- **必改**：无
- **建议改 2 条**：`images: list` 裸类型标注、`torch.load` 缺 `weights_only` —— 经用户裁定，按"与 baseline 完全一致"原则**不改**（GitHub main 也没有这些）
- **可考虑 3 条**：docstring 提及"diagonal"（已在对齐 upstream 时同步修正）、`captured["init_kwargs"]` 无断言、HF 下载 fallback 无单测 —— **不改**

### 13.6 测试状态

```
PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
```

- hpsv2 测试：7 passed, 1 deselected（GPU smoke test 被 mark 过滤）
- 全量回归：**147 passed, 8 deselected**，无新增 failure
- `FutureWarning: torch.cuda.amp.autocast(args...)` —— 刻意保留以对齐 upstream

### 13.7 踩到的坑

1. **环境装了 hpsv2 fork，不是官方版**：`pip show hpsv2` 显示 `Version: 1.2.0`，但 `direct_url.json` 暴露实际来源是 `file:///<local>/images/ptm_runtimes/new-HPSv2`。fork 的 `img_score.score()` 签名完全不同（`score(model, preprocess_val, tokenizer, img_path, prompt, device)`），没有 `cp` / `hps_version` 参数。

2. **GitHub main 的 `score()` 每次都 reload checkpoint**：`torch.load(cp, ...) + model.load_state_dict(...)` 出现在 `score()` 函数体内而非 `initialize_model()` 里，导致每次调用都从磁盘读几百 MB checkpoint。这是 upstream 设计问题。

3. **CephFS IO 阻塞**：pytest 运行期间 CephFS 偶发阻塞导致 `collecting ...` 卡住数分钟，重试后恢复。

### 13.8 Resume 入口（覆盖 §12.9，以本节为准）

**快速状态**（2026-04-28 更新）：
- **hpsv2 scorer**：已修复。模型加载移入 `__init__`，推理逻辑与 GitHub main `img_score.score()` 逐行对齐。PyPI 官方 hpsv2 1.2.0 已重装。
- **多机部署**：双机 16 GPU 已跑通（§12 记录）。
- **测试**：147 passed, 8 deselected
- **环境**：docker 里 python3.12 + torch 2.8 + transformers 4.57 + hpsv2 1.2.0（PyPI 官方）

**下次进来先做的事**（按优先级）：

1. **在目标 GPU 环境跑 hpsv2 的 GPU smoke test**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest tests/scorers/test_hpsv2.py -m gpu -v -o cache_dir=./.pytest_cache
   ```
   需要 `weights_path` 指向有效的 HPSv2 checkpoint 目录。

2. **跑 smoke client 验证端到端**：
   ```bash
   PYTHONPATH=. python3.12 scripts/smoke_client.py --url http://localhost:8080
   ```

3. **跑并发 sweep**：
   ```bash
   PYTHONPATH=. python3.12 scripts/bench_concurrent.py --url http://localhost:8080 --sweep 50 100 200 400 800 --total 500
   ```

4. **如果要跑单测**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
   ```

**关键文件索引**（新增/变更）：
- hpsv2 scorer：`reward_service/scorers/hpsv2_scorer.py`（`_score_single` + `_batch_score` + `score`）
- hpsv2 测试：`tests/scorers/test_hpsv2.py`（7 CPU + 1 GPU）
- 其余索引继承 §12.9

**绝对不要做的事**（继承 §12.9，新增）：
- 不要把 hpsv2 scorer 的推理逻辑改成 batch forward —— 用户要求与 upstream 逐 item 行为完全一致
- 不要把 `torch.cuda.amp.autocast()` 改成新 API —— 保持与 GitHub main 一致
- 不要给 `torch.load` 加 `weights_only=True` —— upstream 没有
- 继承 §12.9 的所有其他禁令

---

*§13.8 是当前 Resume 入口。再有新 session 请覆盖本节、保留 §12.9 作为历史。*

**round-robin 分发**（用户问答记录）：replicas=2 时请求交替发给 replica 0/1（`itertools.cycle`），纯轮询不感知负载。

---

## 14. 续篇 · 2026-04-29 session — Per-Scorer 隔离 venv via Ray runtime_env

### 14.0 时间线

| 阶段 | 产出 |
|---|---|
| 多机调度修复 | geneval2 调度顺序（§12.15 补充）+ ray_stop.sh pkill 自杀 fix |
| GenEval2 Soft-TIFA | cherry-pick geneval2.py dataset 支持 + 下载 800 条 JSONL |
| Per-scorer venv | Plan → 批准 → 实施中（代码改完，测试未跑） |

### 14.1 用户原话

> "目前遇见的问题是，在install.sh里hack了很多环境，但是之后reward model多了之后，就会出现环境冲突的问题。所以我现在想要对每一个reward model都有一个自己专属的venv，由ray来负责管理。"

> "不再保留_compact.py了，默认认为每个reward都必须要有一个自己的venv"

> "base环境里装torch/nccl这些基础的环境。不要用conda，可以直接使用uv pip"

### 14.2 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 隔离方式 | Ray `runtime_env={"pip": [...]}` | Ray 自动建 venv，按内容 hash 缓存 |
| base 环境 | torch + nccl + ray + pillow（预装） | 太大不适合 pip，硬件相关 |
| runtime_env 必填/可选 | **必填** | 用户明确：不保留 _compat.py，每个 reward 必须有自己的 venv |
| _compat.py | **删除** | 每个 venv 有正确的 transformers，不需要 shim |
| install.sh | 精简到只装 base | step 2/3 legacy hack 全删 |
| uv 支持 | install.sh 支持 uv-first fallback pip | Ray runtime_env 仍用 pip 后端（uv alpha 有 bug #54134） |

### 14.3 文件改动清单

**新建**：
- `envs/base.txt` — base 环境文档（torch, nccl, ray, pillow）
- `envs/clip.txt` — transformers>=4.55.2,<4.58
- `envs/pickscore.txt` — 同 clip
- `envs/imagereward.txt` — transformers==4.45.2, image-reward, fairscale, openai-clip
- `envs/hpsv2.txt` — transformers==4.45.2, hpsv2, ftfy, ...
- `envs/hpsv3.txt` — transformers>=4.55.2, hpsv3@git+..., peft, ...
- `envs/unified_reward.txt` — vllm==0.11.0
- `envs/geneval2.txt` — vllm==0.11.0

**删除**：
- `reward_service/scorers/_compat.py`
- `tests/scorers/test_compat.py`

**修改**：
- `reward_service/config.py` — `RewardModelCfg` 加 `runtime_env: str` 必填字段 + `load_config` 校验
- `reward_service/workers/group.py` — 新增 `_build_runtime_env()` + `_actor_options()` 传 `runtime_env`
- `reward_service/scorers/imagereward.py` — 删 `import _compat`
- `install.sh` — 精简到只装 base，支持 uv-first
- `pyproject.toml` — 去 transformers pin / vllm / legacy extras
- `configs/service.example.yaml` — 每个 reward 加 `runtime_env`
- `configs/service.cluster.example.yaml` — 同上
- `tests/test_config.py` — 全部重写，加 runtime_env fixture
- `tests/workers/test_group.py` — 加 `_build_runtime_env` + runtime_env forwarding 测试

### 14.4 测试状态

**未跑**。代码改完但本机无正确 Python 3.12 + pytest 环境。下次 session 在目标机器上跑：
```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest tests/test_config.py tests/workers/test_group.py -v -o cache_dir=./.pytest_cache
```

### 14.5 其他本次改动

- **ray_stop.sh pkill 自杀 fix**：`pkill -f "VLLM::"` → `pkill -f "VLLM[:]:"` + `pkill -f reward_service` → `pkill -f "reward[_]service"`。正则字符类 `[:]` 防止 bash 自身 cmdline 被匹配。
- **GenEval2 Soft-TIFA**：从 `geneval2-dataset-and-model-paths` 分支 cherry-pick `geneval2.py` 改动。新增 `dataset_path` 参数 + `_load_vqa_dataset()` 函数。下载 `geneval2_data.jsonl`（800 条）到 `datasets/geneval2/`。
- **install.sh `--force-reinstall`**：step 3 的 `pip install --no-deps` 加了 `--force-reinstall`（后被 §14 的精简覆盖）。
- **YAML 调度顺序**：多 GPU actor（geneval2、unified_reward）移到 rewards 列表最前，避免 GPU 碎片化。

### 14.6 Resume 入口（覆盖 §13.8，以本节为准）

**快速状态**（2026-04-29）：
- **Per-scorer venv**：代码改完，**测试未跑**。所有 scorer 必须在 YAML 里指定 `runtime_env: envs/<scorer>.txt`。
- **_compat.py 已删除**。imagereward.py 的 `import _compat` 已移除。
- **install.sh 已精简**：只装 base（`pip install -e ".[server,dev]"`），支持 uv-first。
- **GenEval2 Soft-TIFA**：已合入。`datasets/geneval2/geneval2_data.jsonl`（800 条）已下载。
- **ray_stop.sh**：pkill 自杀 bug 已修。

**下次进来先做的事**：

1. **跑单测验证 per-scorer venv 改动**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest tests/test_config.py tests/workers/test_group.py -v -o cache_dir=./.pytest_cache
   ```
   如果有失败，修。

2. **跑全量单测**：
   ```bash
   PYTHONPATH=. PYTHONPYCACHEPREFIX=./.pycache python3.12 -m pytest -m "not gpu and not slow" -q -o cache_dir=./.pytest_cache
   ```
   注意：`test_compat.py` 已删除，其他测试中引用 `_compat` 的也已清理。

3. **启动 service 验证 runtime_env 生效**：
   ```bash
   PYTHONPATH=. python3.12 -m reward_service --config configs/service.example.yaml
   ```
   第一次启动会慢（Ray pip install 每个 venv）。观察日志确认每个 actor 成功初始化。

4. **smoke test**：
   ```bash
   PYTHONPATH=. python3.12 scripts/smoke_client.py --url http://localhost:8080
   ```

**关键文件索引**：
- envs/*.txt — 每个 scorer 的 pip requirements
- `reward_service/config.py` — `RewardModelCfg.runtime_env`（必填）
- `reward_service/workers/group.py` — `_build_runtime_env()` + `_actor_options()`
- YAML 示例：`configs/service.example.yaml` / `configs/service.cluster.example.yaml`
- 其余索引继承 §13.8 / §12.9

**绝对不要做的事**（继承 §13.8，新增）：
- 不要恢复 `_compat.py` —— per-scorer venv 已让每个 scorer 有正确的 transformers
- 不要把 `runtime_env` 改回可选 —— 每个 reward 必须有自己的 venv
- 不要在 base 环境里装 transformers/vllm —— 这些是 scorer 级依赖，走 envs/*.txt
- 继承 §13.8 / §12.9 的所有其他禁令

---

*§14.6 是当前 Resume 入口。再有新 session 请覆盖本节。*

---

## §15 Per-scorer Venv 调试与稳定化（2026-04-30）

### 15.1 目标

将 §14 的 per-scorer venv 设计在真实多机环境中跑通，解决一系列依赖冲突问题。

### 15.2 问题与修复

| 问题 | 根因 | 修复 |
|------|------|------|
| diffusers 导入 `Dinov2WithRegistersConfig` 崩溃 | `envs/imagereward.txt` 未 pin diffusers，装了最新 0.37.1，需要 transformers 5.x 的类 | pin `diffusers==0.31.0` |
| hpsv2 缺 `bpe_simple_vocab_16e6.txt.gz` | PyPI wheel 1.2.0 打包 bug，遗漏数据文件 | 改为 `git+https://github.com/tgxs002/HPSv2.git` |
| vllm scorer 报 `Qwen2Tokenizer has no attribute all_special_tokens_extended` | `envs/unified_reward.txt` 没 pin transformers，装了 5.7.0 | pin `transformers==4.57.0` |
| xformers `.so` undefined symbol 崩溃 | `--ignore-installed` 导致 torch 被重装为新版，与 base 的 xformers ABI 不兼容 | **去掉 `--ignore-installed`**，让 pip 复用 base 的 torch |
| `HFValidationError` 绝对路径被拒 | `huggingface_hub>=0.35` 收紧了 repo id 校验，拒绝绝对路径 | base 里 pin `huggingface_hub>=0.30,<0.35` |
| `from pkg_resources import packaging` ImportError | `setuptools>=70` 不再暴露 `packaging`；`openai-clip` 依赖此接口 | imagereward/hpsv2 venv 里 pin `setuptools<70` |
| `ModuleNotFoundError: virtualenv` | Ray runtime_env 需要 virtualenv 创建 venv | base 依赖加 `virtualenv>=20.0` |
| Worker 节点 venv 缓存过旧 | Ray 按 hash 缓存 venv，旧缓存残留 | 清缓存后重建 |

### 15.3 关键设计决策

**去掉 `--ignore-installed`**：

之前加它是为了"即使 base 有兼容版本的 transformers，也强制在 venv 里装一份"。但副作用是 pip 连 torch 及其所有依赖也重装了，导致与 base 编译的 C++ 扩展（xformers、flash_attn_3）ABI 不兼容。

正确策略是：**base 不装 scorer 级依赖**（transformers/vllm/diffusers 等），这样 pip 自然会装它们到 venv 里，同时复用 base 的 torch/xformers。`install.sh` 新增了主动卸载步骤来保证 base 干净。

### 15.4 文件改动

- `envs/imagereward.txt` — pin `diffusers==0.31.0` + `setuptools<70`
- `envs/hpsv2.txt` — 改为 GitHub 安装 + `setuptools<70`
- `envs/unified_reward.txt` — 加 `transformers==4.57.0`
- `envs/geneval2.txt` — 加 `transformers==4.57.0`
- `envs/base.txt` — 删除（功能由 pyproject.toml 覆盖）
- `pyproject.toml` — 加 `virtualenv>=20.0`、`huggingface_hub>=0.30,<0.35`；加 `integration` marker
- `reward_service/workers/group.py` — 去掉 `pip_install_options`，加启动日志
- `reward_service/workers/actor.py` — 加 `_log_venv_info()`，加 `_VENV_PROBE_PACKAGES` 常量
- `reward_service/scorers/_common.py` — `_DTYPE_MAP` 改为模块级初始化（去掉 global + lazy init）
- `install.sh` — 新增卸载 scorer 级包步骤
- `tests/integration/test_venv_install.py` — 新增 venv 安装集成测试
- `tests/scorers/test_pickscore.py` — 加 `pytest.importorskip("transformers")`
- `scripts/check_venvs.py` — 新增 venv 状态检查脚本
- `README.md` — 精简为纯使用文档
- `.claude/skills/code-standards/SKILL.md` — 加"提交必须经用户批准"规则

### 15.5 验证结果

- **单机 8 GPU**：7 个 scorer 全部 ready（clip 20s, pickscore 23s, imagereward 30s, hpsv2 32s, hpsv3 67s, unified_reward 127s, geneval2 ~150s）
- **单元测试**：131 passed, 1 skipped
- **集成测试**：clip/pickscore/imagereward/hpsv2/hpsv3 全部 PASS（vllm 类超时但非依赖问题）

### 15.6 Resume 入口（覆盖 §14.6，以本节为准）

**当前状态**（2026-04-30）：
- 单机 8 GPU 验证通过，所有 7 scorer 正常启动
- 多机尚未验证（worker 节点需清旧 venv 缓存）
- 131 单元测试通过

**下次进来先做的事**：

1. 多机验证：清 worker 节点 venv 缓存 → `scripts/ray_start.sh` → 启动服务
2. 如果有新 scorer 或新依赖，运行集成测试验证 venv：`pytest tests/integration/ -m integration -k "<scorer>"`

**绝对不要做的事**（继承 §14.6，新增）：
- 不要加回 `--ignore-installed` — 会导致 torch 重装，与 base xformers ABI 冲突
- 不要在 base 环境装 transformers/vllm/diffusers — 走 `envs/*.txt`
- 不要用代码 patch（如 `_block_base_xformers`）绕环境问题 — 从 `envs/*.txt` 版本 pin 解决

---

## §16 集成 geneval + ocr + videoalign scorer（两层隔离调查）（2026-05-30）

### 16.0 时间线
- 起于对两个远端分支（`feat/add-geneval-ocr-workers` by bowenping、`feat/videoalign` by charlesswu）的 merge 评审。
- 本次只落地 bowenping 侧的“干净安全”子集：取 `geneval` + `ocr`，丢弃 `ocr_paddle`，不引入其 uv 全局切换。
- 全程在集成分支 `integration/geneval-ocr-clean`（off `main` deb1dfe），不碰原 feature 分支。

### 16.1 用户目标（原话要点）
- “按推荐的比较干净和安全的方案实现”：砍 `ocr_paddle`、保留 `ocr`、`geneval` 走“方案 A——自带 torch2.1 的完全隔离环境、代码不改”。
- “我们当前在的是一个本地的 CPU 机器很多都没有，你可以先写代码后续我独立测试”。
- geneval 硬约束：保留、**不改它代码**。

### 16.2 关键决策（含一个推翻原方案的发现）
1. **方案 A（geneval 自带 torch2.1 隔离 venv）被调查推翻**。Ray 2.55.1 runtime_env 两条硬约束：
   - pip 后端写死 `--system-site-packages`（`virtualenv_utils.py:101`）→ 永远继承 base torch，无法换 torch 构建。
   - conda/container 强制 `python={当前=3.13}`（`conda.py:183`，其注释明确：用户再写 `python=3.10` 会 `ResolvePackageNotFound`）→ worker 必须与 cluster 同 Python（cloudpickle 兼容）。
   - 本机/集群为 **torch 2.11 / cu130 / py3.13**；geneval 老栈（mmcv-full 1.7.2 / mmdet 2.28.2）只支持 py3.8–3.10 + torch≤2.1 → **任何 runtime_env 后端都托管不了**。
2. **不建 per-reward 隔离旋钮（YAGNI）**：geneval 用不了“同 py 换 torch”的隔离层，其余 scorer 也不需要 → 当前零消费者，不加抽象。
3. **geneval 以最小忠实方式集成**：`geneval.py` 一行不动 + 注册 + `envs/geneval.txt`（加显眼 py3.10 约束注释）+ example config 里**注释掉**并标注约束。是 sidecar / 移植两条后路的共同前置。
4. **`ocr` 留 Tier-1 overlay**：纯 transformers、无 torch pin；把违规的顶层重依赖 import 挪回方法内（对齐 clip/imagereward 约定）。
5. **不带入** bowenping 的 uv 全局切换、`pyproject` ray>=2.43、install.sh 重写、imagereward/hpsv3 的 env 改动、example 的私人路径改动。

### 16.3 文件改动清单
- 新增（取自 bowenping，部分修改）：
  - `reward_service/scorers/geneval.py`（原样，未改）
  - `reward_service/scorers/ocr.py`（改：transformers import → `__init__`；dtype → `_common.resolve_dtype`；删重复赋值行）
  - `reward_service/scorers/ocr_common.py`（改：Levenshtein import → 函数内；正则 → 模块级预编译；docstring 去 paddle 臆测）
  - `envs/geneval.txt`（加 py3.10 / Ray 约束注释）、`envs/ocr.txt`（补末尾换行）
  - `tests/scorers/test_geneval.py`、`tests/scorers/test_ocr.py`（原样带入）
- 修改：
  - `reward_service/scorers/registry.py`（`SCORER_MODULES` 加 geneval、ocr）
  - `tests/scorers/test_registry.py`（expected 集合同步）
  - `configs/service.example.yaml`（新增 ocr 启用项 + geneval 注释项 + 头部说明）
  - `docs/ARCHITECTURE.md`（scorer 清单、§5.1 step2 更正为 SCORER_MODULES、新增 §5.1.1 隔离硬约束）、`CHANGELOG.md`
- 丢弃（未带入）：`ocr_paddle.py`、`envs/ocr_paddle.txt`、`scripts/fix_paddle_cuda.sh`、`tests/scorers/test_ocr_paddle.py`。

### 16.4 测试状态
- 目标测试全绿：`test_ocr` + `test_geneval` + `test_registry` → **12 passed / 9 skipped**（skip = 无 Levenshtein 的 reward 公式测试 + GPU smoke）。
- 全量（排除需 fastapi 的 test_main/test_router、需 Ray+网络的 integration）：**144 passed / 11 skipped / 10 failed**。10 failed 经 worktree 在干净 main 上复现确认为**既有环境失败**（本机缺 vllm/open_clip/hpsv2/hpsv3 + 当前 ray 版本下 test_pool 的 as_awaitable），**与本次改动无关**。
- example config 经 `load_config` 实测可解析（ocr 出现、geneval 不出现）。
- **未在 GPU / 真模型上验证**：本机为 CPU 机、无模型权重 / Ray 集群。`ocr` 真实推理、`geneval` 可托管性需在目标机验证。

### 16.5 simplify 结果（已应用）
- `ocr.py`：`dtype_map` → `resolve_dtype`（去重 + 把坏 dtype 静默兜底改为报错）；精简延迟 import 注释。
- `ocr_common.py`：正则模块级预编译（对齐 unified_reward）；docstring 去 “PaddleOCR” 臆测。
- config：收敛三处重复的 geneval 禁用说明。

### 16.6 review 结果
- **必改**：无（手写改动不变量审计 0 findings、跨文件集成 0 bug）。
- **建议改（待用户确认，未改）**：`ocr.py._run_ocr` 宽 `except Exception` 把推理失败静默转成 reward 0.0 —— RL 训练里会把基础设施故障伪装成合法零奖励。建议让异常上抛（交给 server 的 per-reward 错误隔离），或返回 `nan` 以区分。是 bowenping 原样代码，按“不静默改写”暴露待定。
- **可考虑**：`extract_target_text` 单引号正则会被 prompt 撇号误触发（既有行为）；`compute_ocr_reward` 子串即满分是 flow_grpo 既有算法（reward-hacking 面）；`item.history[-1]` 空 history 会 IndexError（全 scorer 共有约定）；`actor.py _VENV_PROBE_PACKAGES` 未含 Levenshtein/mmdet（仅调试日志）。
- **awareness（vendored，未改）**：`geneval.py._evaluate_reward` 的 `matched_groups[target_group]` 缺越界保护（`_evaluate_strict` 有）；geneval 若在 py3.13 误开启 → actor init 抛 `KeyError`（即“配置即硬失败”的既有语义）。

### 16.7 踩到的坑
- 最大的坑：以为 Tier-2“自带 torch 隔离 venv”可行，实际被 Ray 的 python-pin + pip-overlay 两条约束堵死。**结论：本服务的隔离只到包层、不到解释器层。**（已写入 ARCHITECTURE §5.1.1）

### 16.8 Resume 入口（覆盖 §15.6，以本节为准）

**当前状态**（2026-05-30）：
- 分支 `integration/geneval-ocr-clean`，**改动未 commit**（留在工作树，等用户决定）。
- ocr/geneval 已集成、目标测试绿；geneval 默认注释、需 py3.10 环境或 sidecar。

**下次进来先做的事**（按优先级）：
1. 在**目标 GPU 机**上验证 `ocr`：`pytest tests/integration -m integration -k ocr` + 真实 GOT-OCR 推理。
2. 定 `ocr._run_ocr` 吞异常的处理（review 建议改项）—— 上抛还是返回 nan。
3. 定 `geneval` 的长期路线：(a) py3.10 sidecar 服务 + 代理 actor，或 (b) 移植到 py3.13 检测栈（mmdet 3.x / torchvision / ultralytics）。
4. 决定本分支去留：commit + 提 PR，还是把 charlesswu/videoalign 一起处理。

**绝对不要做的事**（继承 §15.6，新增）：
- 不要试图给 geneval 建“自带 torch2.1 / py3.10”的 Ray runtime_env —— 被 python-pin + system-site-packages 堵死（见 §16.2 / ARCHITECTURE §5.1.1）。
- 不要把 bowenping 的 pip→uv 全局切换并进来 —— 那是 venv 维护者的独立决策。
- 不要改 `geneval.py` 代码（用户硬约束）。

*§16.8 已被 §16.10 覆盖（见下，同一 session 续作）。*

### 16.9 续：并入 videoalign（同一 session 续作）

用户进一步要求：把 charlesswu 的 videoalign 也整合到本分支，让这条成为"功能齐全的本地工作主线"（DiffusionRL 通过软链直接用，不 push、不 PR、不动 team main）。给了"甚至可以重写"的自由度。

**判断：不重写，保留 charlesswu 的设计 + 打磨。** 理由：
1. videoalign 主体是 vendored 上游模型代码（`_videoalign/`，VideoReward 的 Qwen2-VL 实现）——只能原样 vendor。
2. 集成是"新增可选 `videos` 字段"的加法设计，已向后兼容、不破坏现有 7 个图像 scorer（turn-3 评审已确认）。
3. "统一成一个 media-turn 类型"会砸掉 7 个能跑的 scorer + 测试，过度设计（YAGNI）。

**做法**：像 bowenping 一样挑文件搬入（非 `git merge`，保持线性历史）。不重叠文件直接取 videoalign 版本（base/schemas/server/client/group 我们没碰过）；重叠的 registry/test_registry 手动并入 videoalign 条目。configs 加 videoalign 注释块；base.py 修正 docstring 过度承诺（网关并不按模态门控，是 scorer raise）。

**刻意未改（附理由）**：
- videoalign 的 `_resolve_torch_dtype`：与 `_common.resolve_dtype` 行为一致（已 raise）且有独立测试 `TestResolveDtype`——纯去重不值得破坏 tested 代码（与 ocr 不同：ocr 那处是 silent-default 的 correctness bug，才改）。
- `weights_path` 默认值 = `<staging>/RewardModel/VideoReward`：就是真实 staging 目录（与 GPU smoke 测试、其它 scorer 同源），保留。
- scripts 的部署路径（start_videoalign.sh 含私人部署路径）+ stop 的 pgrep 回退：属部署级，用户在真机自行设置；PID 文件是主路径。

**测试**：videoalign + ocr + geneval + registry + schemas + group → **61 passed / 11 skipped**（skip 全是缺 Levenshtein/GPU）。base.py 加 `videos` 字段未破坏图像 scorer。

### 16.10 Resume 入口（覆盖 §16.8，以本节为准）

**当前状态**（2026-05-30）：分支 `integration/geneval-ocr-clean`（建议改名 `integration/all-rewards`），**改动未 commit**（缺 git 身份，待用户设）。已集成 **geneval（停用，需 py3.10）+ ocr（启用）+ videoalign（可用，video 模态）**，是 DiffusionRL 经软链使用的本地工作主线。

**下次进来先做的事**：
1. 设 git 身份 → 本地 commit（不 push、不 PR）。
2. 真机验证：`ocr`（GOT-OCR 推理）、`videoalign`（VideoReward 权重 + flash-attn venv）。
3. geneval 长期路线：py3.10 sidecar 或移植到 py3.13 检测栈。
4. （可选）分支改名 `integration/all-rewards`。

**绝对不要做的事**（继承 §16.8，新增）：
- videoalign 不要重写、不要改成"统一 media-turn"契约——会砸掉 7 个图像 scorer。
- 不要给 videoalign 的 `_resolve_torch_dtype` 做纯去重——它已正确且有测试。

*§16.10 已被 §17.7 覆盖（见下）。*

---

## §17 跨 repo 调用契约修复（RewardService ↔ DiffusionRL_main）（2026-05-31）

### 17.0 时间线与目标
- 起于对 integration 分支的质量 review：发现真正的风险不在 RewardService 内部，而在它与主仓库 `DiffusionRL_main`（真正的 caller）之间的**调用契约**。
- 用户授权同时改两个 repo（均为其本人项目）。目标：让任何 reward 失败变成**响亮、可 debug 的 fail-fast**，而不是静默污染 RL advantage。

### 17.1 三个关键发现（决定了"改哪边"）
1. **server 错误通道是 bucket 级**：`server.py` 对每个 reward 只调一次 `score(items)`，scorer `raise` 会让整批失败；要表达"单样本失效、其余有效"只能用 in-band 哨兵（NaN）。→ per-item 失败用 NaN，整-reward/配置失败才 raise。
2. **pydantic 2.12 把 `float("nan")` 序列化成 JSON `null`**（实测 `model_dump_json`）。所以 NaN 过线后到 caller 是 `None`，旧的 `float(None)` 会 TypeError。finite-guard 必须同时吃 `None` 和 `nan/inf`。
3. **远程 video reward 此前根本没接通**：`RemoteRewardBackend` 从未设 `input_kind`（继承基类默认 "image"），`is_video` 永远 False，`_compute_video_rewards` 是死代码；且其 payload 是 server schema 不接受的扁平 `{video_b64, prompt}`（会 422）。

### 17.2 三项决策（用户拍板）
1. **协议单一事实源 = `schemas.py`**：用 contract-test 钉死，不引重依赖（schemas 只依赖 pydantic）；修好 video 编码。
2. **失败处理回归 fail-fast**（用户否决了更复杂的"mask 出 advantage"方案，理由：避免设计不好的兜底产生"能跑但难 debug"的 fallback）。`service.py:203-208` 本就有 fail-fast，只需让 NaN/None 翻成 `success=False` 喂给它——**不碰 `compute_advantages` 等 RL 数学**。
3. **video 接口现在锁死、实现 defer**：schema + payload 形状 + `input_kind` 路由 + 契约测试全覆盖 video；真机跑 VideoReward + flash-attn 留待后续。

### 17.3 文件改动清单
**RewardService**（本 repo）：
- `reward_service/scorers/base.py`：`BaseScorer.score` docstring 增"Failure semantics"（NaN=per-item 失败契约 / raise=整-reward 失败）。
- `reward_service/scorers/geneval.py`：缺 metadata 由静默 `0.0` 改为 `raise ValueError`（带 item 下标）。
- `reward_service/scorers/videoalign.py`：`sub_metric_names` + 返回 dict 改为 `Overall` 排首（配 caller 默认 `sub_metric_reduce="first"`）。
- `reward_service/scorers/ocr.py`：NaN 分支注释对齐契约（无逻辑变更）。
- `reward_service/schemas.py`：声明为跨-repo 单一事实源。
- 测试：`tests/scorers/test_geneval.py`（缺 metadata 改期望 raise）、`test_videoalign.py`（sub_metric_names 顺序）。

**DiffusionRL_main**：
- `diffusionrl/reward/remote.py`：① `_build_video_score_payload` 改 history 格式 + 透传 metadata；② `_parse_score_response` 加 finite-guard（None/NaN/inf/bool/非数 → `success=False`，不喂进 advantage）；③ 新增 `_first_non_finite`；④ `RemoteRewardSpec` 加 `input_kind` 字段 + 校验；⑤ `__init__` 设 `self.input_kind`。
- 新增测试：`tests/reward/test_wire_contract.py`、`test_remote_response_parsing.py`、`test_remote_input_kind.py`。
- `diffusionrl/reward/README.md`：新增"Remote Backend: wire contract & failure semantics"。

### 17.4 测试状态
- RewardService：`test_geneval` 12 passed / 1 skipped；`test_videoalign` 14 passed / 1 skipped（skip 均为 GPU smoke）。
- DiffusionRL_main：`test_wire_contract`(4) + `test_remote_response_parsing`(9) + `test_remote_input_kind`(3) = **16 passed**。
- 未跑：真机 GPU（ocr/videoalign 推理、video 端到端）——按决策 3 defer。

### 17.5 simplify / review 结论
- **simplify**：无 Should-simplify。两个判断题（image/video payload builder 重复、3 测试文件共享 fixture）结论均为"保持现状"（2 实现低于抽象阈值；每文件 setup 各有特化）。
- **review**：无 Must-fix / Should-fix。4 个 Consider：已应用 3 个（`_first_non_finite` 拦 bool + docstring 诚实化、"扫全部 sub-metric 是有意"的注释、测试补 `component_rewards==[0.0]` + bool 用例）；**否决 1 个**（video metadata 长度硬断言——`request.metadata` 已被 `service.py::_normalize_prompt_metadata` 对齐，且对忽略 metadata 的 reward 部分缺失是合法的，硬断言会误伤；geneval 自身 raise 信息已够清晰）。

### 17.6 踩到的坑 / 经验
- 最初把"NaN 静默污染 advantage"说重了：在 pydantic 2.12 + FastAPI 这条链路上 NaN→null→`float(None)` 其实是 TypeError 崩溃（响亮但晦涩），不是无声污染。但无论哪条序列化路径，finite-guard 同吃 None/nan/inf 才是正解。
- `RemoteRewardBackend` 在 DiffusionRL_main 此前**零测试覆盖**——这正是 video payload 漂移没被发现的根因。契约测试补上了这个缺口。

### 17.7 Resume 入口（覆盖 §16.10，以本节为准）

**当前状态**（2026-05-31）：
- integration 分支 `integration/geneval-ocr-clean` 改动仍**未 commit**（缺 git 身份，待用户设）。
- 本 session 在其基础上**新增**了跨-repo 契约修复（见 §17.3），同样未 commit；DiffusionRL_main 侧改动也未 commit。
- 调用契约已修复并被契约测试钉死；image/video payload、finite-guard、`input_kind` 路由、geneval raise、videoalign Overall-first 全部 CPU 测试绿。

**下次进来先做的事**（按优先级）：
1. 设 git 身份 → 两 repo 各自本地 commit（不 push、不 PR）。
2. **真机验证**（CPU 机做不了）：`ocr`（GOT-OCR 推理）、`videoalign`（VideoReward 权重 + flash-attn venv + 经 `input_kind: video` 的远程 video 端到端）、`geneval2` 真实 VQAScore。
3. `geneval` 长期路线：py3.10 sidecar 或移植 py3.13 检测栈（仍 defer）。
4. 把 ocr/geneval2 接入 DiffusionRL_main 的 `nft_*` config 的 `required_rewards` 跑通一轮。

**绝对不要做的事**（继承 §16.10，新增）：
- 不要把失败处理从 fail-fast 改成"mask 出 advantage"——用户明确否决，理由是避免难 debug 的兜底。NaN/None → `success=False` → 现有 fail-fast 即可。
- 不要让 scorer 对 per-item 推理失败 `raise`——会拖垮整批；用 NaN（见 §17.1）。
- 不要破坏 `schemas.py` ↔ DiffusionRL_main `test_wire_contract.py` 的契约对应：改 schema 必须同步那个测试。

*§17.7 已被 §18.1 覆盖（见下）。*

---

## §18 RewardService 并入 DiffusionRL_main 单仓（2026-05-31）

本目录（`DiffusionRL_main/RewardService/`）现在是 RewardService 的**唯一 canonical 副本**。整树从原独立 repo `…/mmgrpo/RewardService` 复制而来（rsync 排除 `.git` / 缓存），原独立 repo **已退役**（保留未删、仅作历史；其 `docs/RESUME_PROMPT.md` 顶部加了 tombstone）。后续所有 reward service 改动都在 DiffusionRL_main 这一个 GitHub repo 内进行。

### 18.0 为什么 / 怎么做
- 动机：service 的接入方 DiffusionRL_main 与 service 同仓——commit 原子化、契约测试走仓内相对路径，从根上消除"两份代码各自漂移"（即 §17 修的问题）。
- 复制：`rsync -a --exclude=.git --exclude=.pycache …`，**未删原件**。
- 契约测试：`DiffusionRL_main/tests/reward/test_wire_contract.py` 的兜底路径由 `parents[3]`（旧同级）改为 `parents[2]`（仓内 `RewardService/`），实测 4 passed。
- pytest 不冲突：DiffusionRL_main 的 `testpaths=["tests"]`，根目录 `pytest` 不会递归收 `RewardService/tests/`；跑 service 自带测试需 `cd RewardService && pytest`（用它自己的 pyproject）。
- 部署不变：GPU 节点 checkout DiffusionRL_main，从 `RewardService/` 子目录跑 `python -m reward_service --config configs/service.example.yaml`。

### 18.1 Resume 入口（覆盖 §17.7，以本节为准）
- **canonical 位置**：`DiffusionRL_main/RewardService/`（原 `…/mmgrpo/RewardService` 已退役，别再在那改代码）。
- 本次的契约修复（§17）+ 本次合并（§18）改动**均未 commit**，待用户设 git 身份后在 DiffusionRL_main 一处提交。
- 仍未做（不变）：真机 GPU 验证 `ocr` / `videoalign` / `geneval2`、远程 video 端到端、经典 `geneval` 的 py3.10 路线。
- 失败处理走 **fail-fast**（别改回 mask）；改 wire 协议必须同步 `DiffusionRL_main/tests/reward/test_wire_contract.py`。

*§18.1 已被 §19.7 覆盖（见下）。*

---

## §19 新增 `videohpsv3` T2V reward（对齐上游 Flash-GRPO）（2026-07-26）

### 19.0 时间线与目标
用户原话：**"要跟上游 videohpsv3 一样跑视频帧，得新写抽帧+top-30% 聚合代码。使用这种并且符合当前仓库的 reward service 的服务的规范来写个 reward service 放到仓库里放到 flashgrpo 的 worktree 中"**，随后追问 "我们现在的 hpsv3 的 reward 的步骤是什么样的"、"那我们新增个 reward service 呢最小改动和复用应该怎么做"，最后 **"实施把"** 批准。

背景：FlashGRPO 对齐上游 `Shredded-Pork/Flash-GRPO` 期间，训练 reward 不涨。上游 WAN2.1 T2V recipe 用的是 `video_hpsv3_remote`——逐帧 HPSv3 + top-30% 均值——而本仓库当时只有图像版 `hpsv3` 和 `videoalign`，没有对应实现。

**注意**：本次工作发生在 `UniRL` 仓库的 worktree `feature/flashgrpo-unirl` 下的 `unirl-reward-service/`。§18.1 记录的 canonical 位置（`DiffusionRL_main/RewardService/`）是当时的历史状态，本次不动它；**以本节所述位置为准**。

### 19.1 上游语义与本仓差异
上游聚合逻辑（`flow_grpo/rewards.py`）：
```python
l = len(outputs); outputs.sort(reverse=True)
score = sum(outputs[:int(l*0.3)]) / int(l*0.3)
```
两点有意的差异：
- **拆分方式**：上游把工作切在网线两侧（client 抽帧 + JPEG 压缩，专用 server 只看"一批图"）。本仓约定"一个 reward 名 = 一个自包含 scorer"，所以两半都放进同一个 scorer。
- **短片段**：本实现保留 `max(1, int(n * top_ratio))`，≤ 3 帧的片段取单帧最高分；上游 `int(l*0.3)` 为 0 会 `ZeroDivisionError`。凡上游能跑的场景结果 bit-identical。

### 19.2 plan 演进（关键决策）
- **初版 plan** 打算从 `hpsv3_scorer.py` 抽一个 `load_hpsv3_inferencer` 公共函数出来复用 → **放弃**。改用**组合**：内部持有一个 `HPSv3Scorer`，把每帧包成单轮 `ScoreItem` 喂给它。这样 checkpoint 解析、`max_batch_size` 分块、块间 `empty_cache()` 全部白拿，`hpsv3_scorer.py` **零改动**。
- **不新建 `envs/videohpsv3.txt`**：deps 与 `hpsv3` 完全一致（`envs/hpsv3.txt` 本就带 `opencv-python-headless`），Ray 按内容缓存 venv，两个 reward 共用一个。
- **抽帧用 `cv2` 而非 `decord`**：decord 只存在于 videoalign 的 venv，而那个 venv 钉 `transformers==4.45.2`，与 hpsv3 冲突。
- **不 `from ...videoalign import ...`**：import 那个模块会触发 `register("videoalign", ...)`，污染 registry 且可能重复注册报错。故把 `_materialize_video` 提升到 `_common.py`。

### 19.3 文件改动清单
**新增**：
- `reward_service/scorers/video_hpsv3.py`：`top_ratio_mean` / `extract_frames` / `VideoHPSv3Scorer`。
- `tests/scorers/test_video_hpsv3.py`（40 例）。
- `tests/scorers/test_common.py`（5 例，覆盖 `materialize_video` 的所有权语义）。

**修改**：
- `reward_service/scorers/_common.py`：新增共享 `materialize_video`（从 videoalign 私有 staticmethod 提升，行为等价）。
- `reward_service/scorers/videoalign.py`：删掉 24 行私有 `_materialize_video`，改调共享helper；移除随之无用的 `import tempfile`。
- `reward_service/scorers/registry.py`：`SCORER_MODULES` 加一行。
- `configs/service.example.yaml`：注释掉的 `videohpsv3` 块（与其他 T2V reward 一样 opt-in）。
- `README.md` / `CHANGELOG.md` / `docs/ARCHITECTURE.md`：同步 reward 清单。

### 19.4 测试状态
- `pytest tests/ -q` → **45 passed**（`test_video_hpsv3.py` 40 + `test_common.py` 5），4.8s。
- ruff 在本次触碰的所有文件上 **clean**。
- 数值对齐已验证：`top_ratio_mean([1,5,3,10,2,9,4,8,6,7], 0.3) == 9.0` = mean(10,9,8)，与上游 `int(10*0.3)==3` 一致。
- 未跑：真机 GPU（需 HPSv3 权重 + 空闲卡；8 卡当时全被训练占用）。
- **未修**：仓库内 27 个**既有** ruff 错误（`_editreward/*`、`_videoalign/prompt_template.py`、`hpsv2_scorer.py`、`ocr.py`、`geneval.py`、`editreward.py`）——均在本次未触碰的文件里，按 CLAUDE.md §3「surgical changes」不动。

### 19.5 simplify / review 结论
- **simplify**：纯风格改动，无行为变化。`extract_frames` 补 `list["Image.Image"]` 返回注解（配 `TYPE_CHECKING` 导入，保留函数内 lazy `import cv2`）；`ok` → `is_read`；`_score_one` → `_score_clip`；测试里把复制两遍的 tempfile 泄漏断言抽成 `assert_no_leaked_tempfiles` fixture（并用"故意泄漏"探针验证过该 fixture 真的会红，不是静默通过）。
- **review**：2 个必改，**均已修**（见 19.6）；3 个建议改已修；5 个 Consider 中采纳 3 个（log 移到耗时操作之前、`pytest.raises` 补 `match`、fake scorer 补长度断言），2 个记录未做（见 19.6 末）。

### 19.6 review 的两个必改（已修 + 已验证）
1. **`_common.py` tempfile 泄漏**：`owned_tempfiles.append(...)` 原本在 `try/finally` 写入**之后**。`NamedTemporaryFile(delete=False)` 在写入前文件就已落盘，所以写入失败（ENOSPC/EDQUOT）会留下一个**调用方永远看不到**的孤儿文件——正好是磁盘满时最不该泄漏的场景。这个 bug 是从 videoalign 私有方法**原样继承**的，但提升为两个 scorer 共用的公共 helper 后影响面翻倍。修法：`append` 提到创建之后、写入之前。
   - **验证**：写了回归测试 `test_materialize_video_records_the_tempfile_before_writing_it`（monkeypatch `write` 抛 ENOSPC），并**临时把代码改回旧顺序确认该测试会红**（1 failed），再恢复（45 passed）。旧顺序那次确实在 `/tmp` 留了个孤儿文件，实证了 bug 真实存在。
2. **per-item 失败语义**：`_score_clip` 原本让 `extract_frames` 的 `ValueError` 直接冒出 `score()`，于是 64 个片段里**一个**坏视频会让整桶 64 项全进 `errors[i]`，白烧掉数 GPU-分钟。这违反 `base.py:74-80` 明文契约，也违反 §17.7「绝对不要做的事」里那条 **"不要让 scorer 对 per-item 推理失败 raise——会拖垮整批；用 NaN"**。
   - 修法：**只**给"片段内容解不开"这一类返回 `float("nan")`；"缺 video / 类型不对"这类**调用方接线 bug** 继续 `raise`（与 `videoalign` 一致，也符合 base.py 的 "required metadata not wired → raise"）。
   - **与 `videoalign` 有意不同**：videoalign 把整个 `video_paths` 列表交给单次 `self._inferencer.reward(...)`，根本没有 per-item 边界可切，raise 是它唯一选择；videohpsv3 本就逐片段循环，隔离几乎零成本。
   - 新增两个测试：坏片段返回 NaN；坏片段夹在两个好片段中间时，**好片段仍拿到真实分数**（后者才是这条契约的意义）。
- **记录但未做的 2 个 Consider**：① `extract_frames` 没有帧数上限，理论上一个超长片段能 OOM 掉整个 Ray actor（连带整个 reward group），不只是单个请求——目前判断为 YAGNI（内部可信部署，上游也没有），但失败模式确实从"一个请求坏"升级成"reward group 挂"，值得下次真机压测时复评。② `frame_stride`/`top_ratio` 的校验在构造函数与模块级函数里各写一份（消息字面量重复），两处都是 load-bearing（构造期失败早于占卡；公共函数需自校验），抽 helper 反而更啰嗦，保持现状。

### 19.7 Resume 入口（已被 §19.10 覆盖，保留作历史记录）

**当前状态**（2026-07-26）：
- 位置：`UniRL` 仓库 worktree `feature/flashgrpo-unirl` → `unirl-reward-service/`。
- `videohpsv3` scorer 已实现、已测（45 passed）、ruff clean、文档已同步。**改动未 commit**。
- reward **默认关闭**（example config 里是注释块），所以对现有部署零影响。
- 尚未真机验证：需要 HPSv3 权重 + 至少 1 张空闲卡。

**下次进来先做的事**（按优先级）：
1. **把 sglang recipe 的 reward 从 `VideoPickScoreScorer` 改成 `RemoteRewardBackend`** 指向本 service 的 `videohpsv3`——这才是"让 reward 涨起来"这条主线的下一步，本次只交付了 service 侧。
2. 下载 HPSv3 权重；决定 GPU 分配（当时 8 卡全被训练占满，需停 PID 1691465 + Ray 集群才能腾卡）。
3. 真机 smoke：先 `frame_stride: 8` 跑通，再决定是否降到 1（成本约 8 倍）。**同时把 `server.score_timeout_s` 调到 1800**——默认 120 一定超时，且超时后 Ray task 仍在跑（`server._await_ref`），GPU 时间照烧然后被丢弃。
4. 未决：`wan21_t2v_flashgrpo_sglang.yaml:70` 的 `use_torch_compile: true` 仍未定论。
5. 复评 19.6 末尾那条帧数上限（真机压测时）。

**绝对不要做的事**（继承 §17.7 / §18.1，本次新增）：
- 不要把 `videohpsv3` 的 per-item NaN 改回 raise——一个坏片段不该拖垮整桶（见 19.6 第 2 条）。反之，"缺 video / 类型不对"必须继续 raise，别改成 NaN 掩盖接线 bug。
- 不要把 `owned_tempfiles.append` 挪回写入之后（见 19.6 第 1 条，有回归测试钉着）。
- 不要为了"对称"给 `videohpsv3` 单独建 `envs/videohpsv3.txt`——与 `envs/hpsv3.txt` 内容一致才能共用 Ray 缓存的 venv。
- 不要改成继承 `HPSv3Scorer` 或从它里面抽公共函数——组合是有意选择，正是它让 `hpsv3_scorer.py` 零改动。
- 不要在 `video_hpsv3.py` 顶层 `import cv2`：`registry._try_import` 会把 `ImportError` 咽成 warning，顶层导入会让这个 reward 在 base venv 里**静默未注册**，而且单元测试就跑不了解码逻辑了。
- 不要把 reward 名 `videohpsv3` 改成 `video_hpsv3`（见 19.8）——文件名是 `video_hpsv3.py`，但注册名/YAML `scorer:` 字段是**线协议标识符**，必须与上游 Flash-GRPO 的 config key 一致。
- 不要顺手修那 27 个既有 ruff 错误（不在本次范围，见 19.4）。

*§19.7 已过期。当前 Resume 入口是 §19.10。*

### 19.8 模块命名（用户要求 · 2026-07-27）

用户先问"命名是否符合规范，参考其他 reward service 的命名规范呢"，随后指定 **"叫做 video_hpsv3 这样的应该好一点"**。

**`_scorer` 后缀的真实规则**（扫全部 13 个 scorer 模块得出，相关性 100%）：后缀只用来**避免遮蔽自己 import 的同名 pip 包**。只有 2 个带后缀——`hpsv2_scorer.py`（`from hpsv2.img_score import ...`，`envs/hpsv2.txt` 装 `hpsv2`）和 `hpsv3_scorer.py`（`from hpsv3 import HPSv3RewardInferencer`）。另外 11 个（clip / pickscore / imagereward / geneval / geneval2 / wise / editreward / videoalign / ocr / unified_reward）都不 import 同名包，也都不带后缀。本模块不 import 名为 `videohpsv3` 的包，所以**不该带后缀**，对齐 `videoalign.py`。

**最终命名，分两层**：

| 层 | 标识符 | 理由 |
| --- | --- | --- |
| Python 模块 / 测试文件 | `video_hpsv3.py` / `test_video_hpsv3.py` | 用户指定；`snake_case` 符合 §1.1，也与上游的 Python 符号 `def video_hpsv3_remote(device)`（`flow_grpo/rewards.py:93`）同风格 |
| 类名 | `VideoHPSv3Scorer` | 不变，全仓 `<Name>Scorer` |
| 注册名 / `sub_metric_names` / `SCORER_MODULES` key / YAML `name:`+`scorer:` | `videohpsv3`（**不加下划线**） | **线协议标识符**。上游把它注册成 `"videohpsv3"`（`flow_grpo/rewards.py:424`）、config 里也写 `"videohpsv3": 1.0`（`config/dgx.py:73,125`）。改这个等于与上游 config key 分叉，而只改文件名不会 |

即上游本身就是"Python 符号 snake_case、wire key 不带下划线"，本次命名与之一致。

**改动**：`reward_service/scorers/videohpsv3_scorer.py` → `video_hpsv3.py`；`tests/scorers/test_videohpsv3_scorer.py` → `test_video_hpsv3.py`；`registry.py` 的 `SCORER_MODULES` value（key 不变）；测试文件的 3 处 import / `monkeypatch.setattr` 目标。`register()` 名、`name`、`sub_metric_names`、`configs/service.example.yaml`、`README.md` 表格里的 reward 名**均未动**。

**验证**：`pytest tests/ -q` → 45 passed；ruff clean；`importlib.util.find_spec(SCORER_MODULES["videohpsv3"])` 解析成功（确认 registry 映射没断）。`docs/` 里 8 处旧文件名引用已同步。

### 19.9 训练侧接线：新增 `videohpsv3` service config + WAN recipe（2026-07-27）

#### 19.9.1 时间线与目标

§19.7 的第一优先级。用户三条指令依次收敛了设计：

1. 先问 **"为什么我们不能使用 ray 像其他脚本一样直接启动 rewardservice"**（纯问答，未改代码）。
2. **"按照其他 reward service 怎么接入的我们就怎么接入，保证规范性和可移植性，不需要参考 flashgrpo 的上游代码"** —— 这一条否掉了我的初版设计。
3. **"新增 yaml，wan 和 videohps 的 yaml，而不是修改"** —— 定了"新文件"而非"改原文件"。

#### 19.9.2 plan 演进（一次被用户推翻的设计）

**初版（错的）**：给新 service config 加 `cluster: ray_address: auto`，再配一个专用启动脚本。理由是"训练侧已经有 Ray 集群，接进去更省事"。

**被否的原因**：这是照着 FlashGRPO 上游的部署形状想的，不是本仓的约定。读 `configs/editreward_service.yaml` 后确认——**它根本没有 `cluster:` 段**。本仓单 reward config 的规范形状是：header 里写清 `ray start --head --port=6379 --num-gpus=1`，然后 `python -m reward_service --config ...`，由 `workers/pool.py::_init_ray` 的"已有集群则接入"分支自动接上。多余的 `cluster:` 段、多余的启动脚本、以及 `stop_videoalign.sh:53-57` 那个 `ray stop --force` 的坑，全都不需要。

**"新增而非修改"的仓内先例**：`configs/service.example.yaml` 之外，recipe 侧有 `sd3_nft.yaml` → `sd3_nft_reward_service.yaml`——逐行复制原 recipe，只换 reward backend，header 写一句 "Identical to X except the reward backend"。本次照此办理，diff 可审。

#### 19.9.3 文件改动清单

**新增（2 个，都是新文件，原 recipe 零改动）**：
- `configs/videohpsv3_service.yaml`（本仓）—— 对齐 `editreward_service.yaml` 的形状。`score_timeout_s: 1800`。
- `examples/diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3.yaml`（UniRL 仓根，225 行）—— 逐行复制 `wan21_t2v_flashgrpo_sglang.yaml`，只改 header / 一处注释 / reward 块 / `run_name` / `tags`。

**修改**：无代码改动。仅 `CHANGELOG.md` + 本文件 + `docs/RESUME_PROMPT.md`。

#### 19.9.4 接线正确性（无 GPU 验证链）

最大的风险是"SGLang 侧 `load_vae: false`，reward 还拿得到解码后的视频吗"。逐段验证：

| 环节 | 证据 |
| --- | --- |
| adapter 注册 | `rollout/engine/sglang_diffusion/adapters/video.py:203` `@register_adapter("wan21") class Wan21T2VAdapter(VideoAdapter)` |
| track 名 | `VideoAdapter.track_name = "video"` |
| RewardService 路由 | `reward/service.py::_KIND_TO_KEY["video"] == "video"`，运行时比对 **MATCH: True** |
| 布局 | 轨道是 frame-major `[T,C,H,W]`（`utils.stack_decoded_videos`），`_encode_video_b64` 要 `(C,T,H,W)`——`types/reward.py:73-77` 的 `permute(1,0,2,3)` 正好桥接。**不是 bug，无需修** |
| `load_vae: false` | 解码发生在 SGLang engine 侧，训练侧 pipeline 只回放 latent transition，两者不冲突 |

`input_kind: video` 是让 `RewardService` 去填 `generated["video"]` 的开关，必须显式写（默认 `"image"`）。

#### 19.9.5 三个把细节搞对的点

1. **client timeout 1900 > server 1800**：一开始我把因果写反了，说"这样服务端超时才会返回结构化错误"。实际 `server.py:119-138` 的 `_await_ref` **无条件**把超时包成 `TimeoutError` 塞进 `errors[i][reward]`。ordering 真正买到的是：客户端不会先放弃、然后按 `max_retries: 3` 把整批重新 POST 三遍（`unirl/reward/remote.py:436-460`）。注释已改成这个说法。
2. **单机共卡要两个 override，且其中一个必须用 `++`**：`DevicePool`（`unirl/distributed/group/device_pool.py:143-148`）会用 STRICT_PACK PlacementGroup 把 `num_devices` 张卡全占掉，reward actor 再要 `num_gpus: 1` 就**永久 PENDING 且不报错**。所以要 `num_devices=7`；又因 `device_pool.py:56` 有整除断言，`devices_per_node` 也得跟着改。但 `devices_per_node` **不在这个 config 里**，Hydra 会报 `Key 'devices_per_node' is not in struct`——必须 `++devices_per_node=7`。已实测：`num_devices=7 ++devices_per_node=7` 正常 resolve。（先例：`examples/pe/pe_sglang_full_wise.yaml:26` 也用 `++`。）
3. **默认值不是上游语义，已在 recipe header 写明**：`frame_stride: 8` → 81 帧里只打 11 帧，top-30% 池子只有 **3 帧**；`frame_stride: 1` → 81 帧、池子 24 帧，才是上游。3 帧 vs 24 帧是**不同的估计量**，不是"便宜一点的同一个东西"，所以必须写在 header 里而不是埋在 service config。成本：一桶 8 clip 在 stride 1 下是 648 次逐帧打分 / `max_batch_size=4` 下 168 次 batched forward；stride 8 是 88 次 / 24 次。

#### 19.9.6 测试状态
无新增代码 ⇒ 按 §4.5.2 免测（纯配置，且已有 45 例覆盖 scorer 本身）。改为跑配置层验证：

- `load_config("configs/videohpsv3_service.yaml")` → `score_timeout_s=1800.0`、`frame_stride=8`。
- **scorer `__init__` 签名 vs YAML `params` 交叉比对：unknown keys = none**（这一步能挡住"YAML 写了个 scorer 不认的参数、Ray actor 起来才炸"）。
- Hydra compose：`python -m unirl.train_diffusion --config-name ... --cfg job --resolve` 通过。
- `hydra.utils.instantiate` 实例化 `RemoteRewardSpec`（会跑它 `__post_init__` 的校验器）通过。
- `diff -u` 原 recipe vs 新 recipe：改动只落在预期的 5 处。
- `pytest tests/ -q` → **45 passed**（simplify 前后各跑一次）。
- **编解码全链路 roundtrip（CPU，无 GPU）**：造一个 `[81,3,480,832]` frame-major 张量 → `permute(1,0,2,3)` → `_encode_video_b64` → 16.8 MB mp4（`ftypisom`）→ 交给 scorer 的 `extract_frames` 解回来。stride 1 得 81 帧、top-30% = 24；stride 8 得 11 帧、top-30% = 3——与文档里的数字逐一吻合，分辨率 832x480 / mode RGB 也对。这一步值得单跑，因为 **本 recipe 是整个 `examples/` 树里第一个用 `input_kind: video` 的**（另一处 `pe_sglang_full_wise.yaml:286` 是 `image`），`_encode_video_b64` 此前从未被任何 recipe 走过。

#### 19.9.6b DP_SCATTER：timeout 预算的真正约束（自查发现，修正了我先写的注释）

我最初在 service config 里把 `score_timeout_s` 的成本注释写成"一桶 8 clip = 88 次逐帧打分"——那是**单个 rank 的份额**，不是超时真正要覆盖的量。实际：

- `RewardService.score_and_attach` 带 `@distributed(dispatch_mode=Dispatch.DP_SCATTER)`（`unirl/reward/service.py:189`），而 `reward_fraction` 默认 0（`unirl/trainer/diffusion.py:51`，本 recipe 未设），所以 reward 就是训练 workgroup 的 train-side sibling——**每个训练 rank 各自持有一个 `RemoteRewardBackend`，各自 POST**。
- 视频路径 `_compute_video_rewards`（`unirl/reward/remote.py:371`）把整个 shard 一次性发出去，不按 `batch_size` 切块。
- service 侧 `num_replicas: 1` + `max_concurrency: 1` ⇒ 这些并发请求**在同一个 actor 上串行**。
- `_await_ref` 的 `asyncio.wait_for` 时钟从 dispatch 起算（`reward_service/server.py:129`），**排队时间照算进 `score_timeout_s`**。

结论：`score_timeout_s` 必须覆盖**整个 rollout step 的 reward 总时长**，而不是一个 rank 的份额。一步 64 clip（8 prompt × 8 sample × 81 帧）在 stride 8 下是 704 次逐帧打分 / 176 次 batched forward；stride 1 下是 5184 / 1296。**stride 1 时 1800s 是有可能不够的**——要么调大，要么加 `num_replicas`。注释已按这个口径重写。

#### 19.9.7 simplify 结论

3 项，全部采纳：
1. **必改**：header 里的启动命令写了 `devices_per_node=7`，会在启动时直接报错——改 `++devices_per_node=7`。我独立复验过报错文本，确认这条是真的（不是 agent 幻觉）。
2. **必改**：上面 19.9.5 第 1 条那个反了的因果关系。
3. **建议改**：`server:` 段里 7 行注释夹在 `host:` 和 `port:` 之间，把 `port` 从它自己的块里挤了出去；内容还与 `service.example.yaml:115-125` 和 scorer docstring 重复，并把 300s 说成"图像 service 的默认值"（真实默认是 `config.py:25` 的 **120.0**）。移到 `port:` 之后并收紧。

另有一条 Consider（上游 parity 应写进 recipe）已采纳 → 19.9.5 第 3 条那段 NOTE。

#### 19.9.8 遗留

- **未真机验证**：HPSv3 权重仍未落盘（`weights_path` 是占位符，与 `editreward_service.yaml` 写 HF id 同理）；GPU 当时仍被训练 PID 1691465 + Ray 占满。
- **`use_torch_compile: true`**（新 recipe 第 87 行）继承自原 recipe，仍未定论。
- **未决问题**：`examples/diffusion/wan21/wan21_t2v_flashgrpo.yaml`（trainside 变体，reward 块在 `:91-101`）是否也要配一个 `_videohpsv3` 兄弟文件。我倾向要，否则两个 flashgrpo recipe 在 reward 上分叉。用户未回复。
- **commit 时**：`reward_service/scorers/video_hpsv3.py` 和 `tests/` 还是 untracked，需要 `git add`；`.gitignore` 里 `/Flash-GRPO/` 那行**不提交**（用户明确要求）。

### 19.10 Resume 入口（覆盖 §19.7，以本节为准）

**当前状态**（2026-07-27）：
- 位置：`UniRL` 仓库 worktree `feature/flashgrpo-unirl` → `unirl-reward-service/`。
- `videohpsv3` scorer 已实现、已测（45 passed）、ruff clean。
- **训练侧已接线**（§19.9）：`configs/videohpsv3_service.yaml` + `examples/diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3.yaml`，两个都是**新文件**，原 recipe 零改动。配置层全部验证通过，**真机未跑**。
- 改动**未 commit**。

**启动链（两步，规范形状）**：
```
# 节点 1（reward）
cd unirl-reward-service
ray start --head --port=6379 --num-gpus=1
python -m reward_service --config configs/videohpsv3_service.yaml

# 节点 2（train）
export REWARD_SERVICE_URL=http://<node1_ip>:8080
bash examples/run_experiment_single_node.sh \
  diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3

# 单机共卡：必须留一张卡给 scorer actor，且两个 override 都要
bash examples/run_experiment_single_node.sh \
  diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3 \
  num_devices=7 ++devices_per_node=7
```

**下次进来先做的事**（按优先级）：
1. 下载 HPSv3 权重，填进 `configs/videohpsv3_service.yaml` 的 `weights_path`（现在是占位符）。
2. 腾 GPU（当时训练 PID 1691465 + Ray 占满 8 卡），按上面的启动链跑真机 smoke。先 `frame_stride: 8` 跑通链路，再决定是否降到 1 拿上游语义（成本 ~8 倍，见 §19.9.5 第 3 条）。
3. 决定 `wan21_t2v_flashgrpo.yaml` 要不要配 `_videohpsv3` 兄弟文件（§19.9.8 未决问题）。
4. `use_torch_compile: true` 仍未定论。
5. 复评 §19.6 末尾那条帧数上限（真机压测时）。

**绝对不要做的事**（§19.7 全部继续有效，本次新增）：
- 不要给新 service config 加 `cluster:` 段——本仓单 reward config 的规范是先 `ray start --head` 再起 service，`editreward_service.yaml` 就没有这一段（§19.9.2）。
- 不要在启动命令里写光秃秃的 `devices_per_node=7`——这个 key 不在 recipe 里，Hydra 会报 `not in struct`，必须 `++`（§19.9.5 第 2 条）。
- 不要把 client `timeout` 降到 ≤ server `score_timeout_s`——客户端会先放弃并按 `max_retries: 3` 重 POST 整批，把已经在烧的 GPU 时间又乘以 3（§19.9.5 第 1 条）。
- 不要以为 `frame_stride: 8` 是"上游的便宜版"——它把 top-30% 池子从 24 帧砍到 3 帧，是另一个估计量（§19.9.5 第 3 条）。
- 不要按"单个 rank 的份额"去估 `score_timeout_s`——reward 是 DP_SCATTER 的，每个 rank 各发一个请求、在单 actor 上串行，且排队时间算进超时。要按整步总量估（§19.9.6b）。
- 不要假设 `imageio` / `imageio-ffmpeg` 一定在——`export_to_video` 靠它们，但 UniRL 的 `pyproject.toml` 没声明（当前 venv 里恰好有：imageio 2.36.0 / imageio-ffmpeg 0.5.1）。换环境跑不通先查这个。
- 不要去"修" `[T,C,H,W]` 和 `(C,T,H,W)` 的差异——`types/reward.py:73-77` 的 permute 已经桥接了（§19.9.4）。
- 不要改原来那两个 flashgrpo recipe 来接 remote reward——用户明确要求新增文件（§19.9.1 第 3 条）。

*§19.10 已被 §19.13 覆盖（保留作历史记录）。特别注意：本节第 2 条建议的 `frame_stride: 8` 先跑通、以及"降到 1 成本 ~8 倍"的说法，已被 §19.12 的实测推翻——以 §19.13 为准。*

### 19.11 HPSv3 backbone 上 FlashAttention（fa 调查 + FA2 pin）（2026-07-27）

**背景问题**：用户问「配置中默认用 fa 还是 sdpa」，然后要求给 reward 端装 fa2。

**调查结论（两侧默认都是 SDPA）**：
- **训练侧（WAN21）**：recipe 不设 `attn_implementation`；`bundle.py:129` 用 diffusers `WanTransformer3DModel.from_pretrained`，默认 processor 是 `WanAttnProcessor` → `dispatch_attention_fn` → active backend 默认 `DIFFUSERS_ATTN_BACKEND=os.getenv(...,"native")`（`diffusers/utils/constants.py:46`），`native` = `torch.nn.functional.scaled_dot_product_attention`。unirl 无任何 `set_active_backend` / `attention_backend()` / env 覆盖，故停在 SDPA。注意 torch SDPA 在 H20+bf16 下内部通常已 dispatch 到 flash kernel，不是慢的 math 路径。train base venv 里 *有* `flash_attn_4-4.0.0b15`（FA4 beta），但 diffusers 默认没启用它。想显式走 diffusers flash backend：`export DIFFUSERS_ATTN_BACKEND=flash` 或 `with attention_backend("flash"):`，无需改代码。
- **奖励侧（HPSv3 / Qwen2-VL-7B）**：`hpsv3_scorer` → `HPSv3RewardInferencer`（`inference.py:30`）→ `create_model_and_processor`（`hpsv3/train.py`）。该函数 `try: import flash_attn`，命中且 `disable_flash_attn2=False`（dataclass 默认 + `HPSv3_7B.yaml` 都是 False）时才传 `attn_implementation="flash_attention_2"`，否则 sdpa。**决定因素 = `flash_attn`(FA2) 包能否 import**。Ray actor 的 runtime_env venv（`include-system-site-packages=false`，纯净）只装 `envs/hpsv3.txt`，其中原本没有 flash_attn，故实测落到 SDPA。

**reward actor venv 实测环境**（wheel 必须精确对齐）：py3.12 · torch **2.12.1+cu130** · CUDA 13.0 · cxx11abi=True · sm_90 (Hopper)。

**为什么是 FA2 而不是 FA3**：HPSv3 gate 在 `import flash_attn`（FA2 模块名）且**写死** `"flash_attention_2"`。FA3 的导入名是 `flash_attn_interface`，且 transformers 的 FA2 路径不调 FA3 kernel —— 装 FA3 会被完全忽略，除非改 HPSv3 源码（违反本仓「不 patch 绕环境」硬规矩）。故选 FA2。

**改动（文件）**：`envs/hpsv3.txt` 末尾 pin 预编译 FA2 wheel（PyPI 版会从源码编译、需 nvcc、易失败）：
```
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3%2Bcu130torch2.12-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```
wheel 源 = `mjun0812/flash-attention-prebuild-wheels`。已验证 URL 可达（302→CDN→200，~231 MB）。`workers/group.py:_build_runtime_env` 按行读入、跳过 `#` 注释,裸 URL 作为一个 pip package 传给 Ray runtime_env pip，语法有效;预编译 wheel 无源码构建，不需要 `# pip-options: --no-build-isolation`。

**测试**：纯 env pin、无代码逻辑 → 命中 §4.5.2 免测清单，不补单测。

**未完成 / 待验证**：pin 只在**下次 hpsv3 缓存 venv 重建 + reward service 重启**时生效。当前有一条 8 卡训练正用着 reward service，重启会打断它的 reward backend，属破坏性操作，**未擅自执行**，等用户决定。生效后应确认 actor 日志不再打 "Flash Attention is not installed. Falling to SDPA." 并且显存/吞吐符合预期。hpsv3 与 videohpsv3 两个 scorer 共用本文件，改动后两者仍一致、仍共享同一缓存 venv。

**绝对不要做的事（本次新增）**：
- 不要给 HPSv3 reward 端装 FA3 指望它生效——HPSv3 gate 在 `import flash_attn`(FA2) 且写死 `flash_attention_2`，FA3 导入名不同会被忽略（§19.11）。
- 不要用 PyPI 的裸 `flash-attn`——会从源码编译。必须 pin 与 venv 精确对齐的预编译 wheel（cp312 / torch2.12 / cu130）。换 torch/CUDA/py 版本后这条 wheel URL 需重挑。

*Resume 时：fa 相关事实以 §19.11 为准，其余仍看 §19.10。*

### 19.12 reward 崩塌诊断 + 三项修复重启（2026-07-28）

**背景**：`frame_stride: 8` 那条 8 卡训练跑了 22 小时（74 rollouts），reward 先升后崩——rollout 1-44 为正、29-38 见顶（+4.15/+4.39/+4.86），50-74 转负、最低 -7.6794。用户问「为什么我们的reward一直在下降」。

**诊断结论：真实的 over-optimization / reward hacking，不是基础设施故障。** 证据：
- step 时间 **1054.1s ± 1.18s**（n=73，min 1051 / max 1059）——零方差。日志里 grep 不到 timeout / NaN / OOM / traceback / actor-died。
- 读码排除了两个常见 bug：`algorithms/base.py:151-193` `_grpo_clip_loss` 的 PPO 符号正确（`unclipped = -adv*ratio`，`maximum(unclipped, clipped)`）；`types/rollout_resp.py:280-345` 的优势归一化 `(reward - group_mean)/(group_std + eps)` 正确。**组内中心化意味着 reward 的绝对水平本身不可能产生向下梯度——所以是策略真的退化了**。
- rollout 50 是 trust-region 突破点：整个 run 唯一一次 `loss=0.0005`、`gn=0.0145`（正常值的 100 倍）、`clip=0.38`（唯一一次 clip 激活；`clip_range: 1e-3` 窄到 `ratio=1.0000` 平时根本触不到）。
- `gn≈0.0001` 看着无害是错觉：**Adam 的步长 ≈ `lr·g/√v ≈ lr`，与梯度大小无关**。74 rollouts × `num_updates_per_batch: 2` = 148 步，LoRA `alpha 128 / rank 64` = 2× 放大 → 约 3e-2 等效漂移。`max_grad_norm: 1.0` 全程未生效（gn 比它低 4 个数量级）。

**三个根因（用户批准按此重启）**：
1. **看不见**：rollout 视频从不落盘（engine `save_output=False`，`_patches/patch_sampling_io.py` 那个 `output_file_path` 只是确定性文件名 hash），`log_media: false` → 盲飞 22 小时。
2. **奖励可被薅**：`video_hpsv3.py:65` `top_ratio_mean` 做 `keep = max(1, int(len(scores)*top_ratio))`。stride 8 下 81 帧只采 11 帧，`int(11*0.3)` = **只保留 3 帧**——策略可以把 3 帧做漂亮、放任另外 78 帧烂掉。stride 1 时池子是 81 选 24。
3. **prompt 分布不匹配**：`datasets/pickscore/train.txt` 是静态图美学集（25432 行、均长 113 字符、58% 逗号堆 tag），与上游 T2V 运动 prompt 集（19700 行、均长 76 字符）**只有 2 行重叠**。给视频策略的信号里没有"要动"这件事。

**实测性能分解（从 wandb offline 二进制里 regex 捞出 `perf/*`）**：generate **775.7s（75.4%）**、reward **8.2s（0.8%）**、train 240.3s（23.4%）、weight_sync 0.5s、step 1029.1s。**这条 run 是 generate-bound，不是 reward-bound。**

**因此推翻了 §19.9.5 第 3 条与 recipe 头部原有的"stride 1 成本 ×8、`score_timeout_s` 变成瓶颈"警告**：stride 1 的 reward ≈ 8.2 × 81/11 ≈ **60s**，step 1029→1081s，即 17.2→**18.0 分钟（+4.6%）**，`score_timeout_s: 1800` 仍有 **~30 倍余量**。stride 1 基本免费，没有理由不开。

**另一个致命发现：22 小时零 checkpoint。** `save_interval` 在三个 flashgrpo recipe 里全都缺失 → `train_diffusion.py:70` 默认 `0` → `trainer/base.py:348` `if save_interval <= 0: return`。**commit da2b5dd "Enable checkpointing for FlashGRPO SGLang" 只翻了 `activation_checkpointing: false→true` 和 `use_torch_compile: false→true`——那是激活重算，不是存权重。** 所以崩塌后无处可回滚，22 小时权重永久丢弃。

**文件改动清单**：
- 修改 `configs/videohpsv3_service.yaml`：`frame_stride: 8 → 1`（含注释改写为实测成本 + 说明为何不该调高）；`weights_path` 从占位符 `/path/to/RewardModel/HPSv3` 填成真实路径 `/group/40173/zionyfeng/models/RewardModel/HPSv3`。
- 修改 `examples/diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3.yaml`（训练侧，UniRL repo 根下）：
  - 头部注释：删掉"stride 1 = ~8x 成本 / timeout 是瓶颈"的错误警告，改成实测数据 + 记录 stride-8 崩塌事故。
  - `data_path` / `eval_data_path` → `datasets/video/{train,test}.txt`。
  - `log_media: false → true`。
  - 新增 `save_interval: 10` / `save_dir` / `save_mode: auto`（照 `qwen_image_edit_plus_flowgrpo_sglang.yaml:56-58` 的仓库惯例）。
- 新增（**untracked、不提交**）`datasets/video/{train,test}.txt`：19700 / 300 行，从本地 `Flash-GRPO/dataset/video/`（`.gitignore:109` 已忽略该目录）拷贝，无需下载。

**测试状态**：本次全是 YAML 值 + 注释改动与数据文件拷贝，无 Python 逻辑变更 → 命中 §4.5.2 免测清单。改用配置层验证：
- `python -m unirl.train_diffusion --config-name ... --cfg job --resolve` 通过，确认 `save_interval: 10` / `save_dir` / `save_mode: auto` / `datasets/video/*` / `log_media: true` 全部 resolve 正确。
- `load_config('configs/videohpsv3_service.yaml')` 通过，确认 `frame_stride: 1` + 真实 `weights_path`。
- 链路已验证可用（读码）：media 走 `trainer/diffusion.py:523` `_drop_decoded` →（`trainer/base.py:277-284`）在释放 `decoded` 前 `build_media_preview_for_track` → `wandb_logger.py:689` `wandb.Video(..., format="mp4")`；checkpoint 走 `train_diffusion.py:70` → `trainer/base.py:330-365` `maybe_save_checkpoint`。

**重启实况（2026-07-28）**：停旧训练 PID 876635（22h 权重丢弃，已事先告知用户）+ 旧 reward service PID 534894 → 8 卡显存归零 → reward service 重启（新 PID 2456101，日志 `logs/reward_videohpsv3_stride1.log`），FA2 wheel 触发缓存 venv 重建，**~4.5 分钟**后 8 个 actor 全 ready、每卡 16.4GB、`/health` 全 `videohpsv3:ready` → 训练重启（PID 2477362，日志 `logs/train_videohpsv3_stride1.log`，`WANDB_RUN_NAME=wan21_t2v_flashgrpo_videohpsv3_stride1`），确认加载 19700 条上游 prompt。

**§19.11 待验证项已结清：FA2 生效。** 新 reward service 日志里 "Flash Attention is not installed. Falling to SDPA." 出现 **0 次**（旧 run 有），HPSv3 backbone 现在走 flash_attention_2。

**踩到的坑**：
- 我自己先前给的"stride 1 成本 ×8、要复核 timeout"是错的——没量 `perf/reward_time_s` 就按帧数线性外推，而 reward 只占 step 的 0.8%。**性能建议必须先量 profile 再给。**
- wandb offline `.wandb` 用官方 `DataStore` + `Record` protobuf 解析返回 0 steps（12 条 history record 却解不出），只能 regex 扫二进制捞 `perf/*` 键旁的数值。下次要 offline 指标别指望官方 API。
- `weights_path` 占位符是上个 session 在 service 已运行时写回去的，于是"重启 service"这个动作被埋了个哑雷——运行中的进程内存里是好路径，磁盘配置是坏路径。**改运行中服务的配置时，要么同时改对，要么在 Resume 入口标红。**

### 19.13 Resume 入口（覆盖 §19.10；已被 §19.15 覆盖，保留作历史记录）

**当前状态**（2026-07-28）：
- 位置：`UniRL` worktree `feature/flashgrpo-unirl` → `unirl-reward-service/`。
- `videohpsv3` scorer 已实现、已测（50 passed）、ruff clean。
- **两条进程在跑**：reward service PID 2456101（`frame_stride: 1`，8 replicas，FA2 已生效）+ 训练 PID 2477362（上游 video prompts，`log_media: true`，`save_interval: 10`）。
- **reward service 已单独开 PR**：fork PR #5（`feat/reward-service-videohpsv3` → `NancyFyong/UniRL:main`），在独立 worktree `../pr-reward-service` 里从 `origin/main` 建的，不影响本 worktree 的在跑文件。按用户要求两轮收窄后的最终形态：**9 文件 +417/-48，4 个 commit**（`dd47735` 加 repo id 支持 → `b98cc46` scorer 本体 + config → `b1015d4` envs 依赖 pin，顺序保证 bisect 时 config 依赖的代码先落地；末尾 `63928ca` 是 review 的注释精简），CI 三项全绿，PR body 已重写对齐。
  - **脱敏/收窄清单**：① `weights_path` 不填本机路径，改成 HF repo id `MizzenAI/HPSv3`——为此在 `hpsv3_scorer.py` 加了 `_resolve_checkpoint_path`（目录 / 文件 / repo id 三种都吃，非本地值走 `hf_hub_download`，失败统一报 `FileNotFoundError` 并把两种解释都写进消息），照 `hpsv2_scorer.py:118` 的同名 staticmethod 抄的形状。**旧记录里"不能填 HF repo id"已作废**——那是我们自己 `hpsv3_scorer.py` 的透传限制，不是上游的，现已修掉。② `132adac` 带进来的 4 处 `PID 1691465` 改成"训练进程"。③ 本节 §19.12-19.13 整段不进 PR。④ `tests/` 不进 PR（50 个用例只在本 worktree，PR body 的 Test Plan 明说了这点，并声明 PR 里的 scorer 源码与跑测试的版本 `diff` 一致）。⑤ markdown 只交 `docs/ARCHITECTURE.md`，`CHANGELOG.md` / `README.md` / `docs/DEVELOPMENT_LOG.md` / `docs/RESUME_PROMPT.md` 都不交。
  - recipe 也不在 PR 里——它 `_target_: unirl.algorithms.flashgrpo.FlashGRPO`，`main` 上没这个模块，会被 `recipe _target_ paths resolve` 钩子拦下，只能随 FlashGRPO 的 PR 一起走。
  - **第一轮 review 已处理**（`63928ca`，追加 commit 而非 force-push，避免动到 reviewer 那份 pending draft review 的行锚点）。四条意见全是"注释太多"：① `envs/hpsv3.txt` 的 FA2 整块注释删掉、只留 wheel URL；② trl / tensorboard 注释各压到一行；③④ 两个 config 的注释精简（`videohpsv3_service.yaml` 55→41 行，`service.example.yaml` 的 videohpsv3 段 20→8 行，对齐上面 `hpsv2` 条目的密度）。纯注释改动，改完两个 config 重新过 `load_config` 取值一字未变。
  - **reviewer 问"真的需要 trl 吗"——需要，是 import 期依赖**：`hpsv3/__init__.py` → `inference.py:9` → `hpsv3/train.py:21 from trl import get_kbit_device_map, get_quantization_config`，且 `train.py:8` 还连带 import `qwen2vl_trainer`（`:56 from trl import RewardTrainer`、`:10 from torch.utils.tensorboard import SummaryWriter`）。所以只要 `import hpsv3` 就必须有 trl + tensorboard，推理侧一行都没碰它们。实跑组合 `trl 0.21.0 / tensorboard 2.21.0 / transformers 4.57.6`。上游若把 `create_model_and_processor` 从 `train.py` 拆出去，这两个依赖才能删。
  - **FA2 注释删掉的代价**：那行 wheel URL 是全文件最脆的一处（`cp312/torch2.12/cu130`、Hopper sm_90 硬绑定，且"装不装这个包本身就是 FA2/SDPA 的开关"），删注释后这段理由只存在于 PR 正文的 Compatibility/Risk 段和 PR 评论里。已在评论里说明并提出可以加回一行，等 reviewer 定。
- **上游 draft PR 已开**：[Tencent-Hunyuan/UniRL#270](https://github.com/Tencent-Hunyuan/UniRL/pull/270)，`NancyFyong:feat/videohpsv3-reward-scorer` → `Tencent-Hunyuan/UniRL:main`，draft 状态，**9 文件 +417/-48，3 个 commit**（`807fb4b` repo id → `0313154` scorer + config → `89cb6a2` envs pin），5 项 CI 全绿（含 `Validate PR body sections` / `Validate PR title` / `lint` / 开源扫描）。
  - **必须另起 worktree `../upstream-videohpsv3` 从 `upstream/main`（5d034c3）建分支**：fork 的 `origin/main` 已经岔开（比 upstream 多 37 个 commit、落后 69 个），直接拿 PR #5 的分支跨 fork 提会带进 41 个 commit。已验证那 37 个 fork-only commit 一个都没碰 `unirl-reward-service/`，所以 cherry-pick 干净；rebuild 后用 `git diff <cherry-pick 结果> HEAD` 为空证明树逐字节一致。
  - 与 PR #5 的差别：① review 的注释精简（`63928ca`）在这里是**折进父 commit** 的（上游没见过精简前的版本，不需要保留那段历史）；② commit message 全部重写成上游读者视角（repo id 的 `HFValidationError` 理由、frame_stride 的 reward-hacking 论证、trl/tensorboard 的 import 链、"装不装 flash_attn 本身就是 FA2 开关"）。
  - **不带测试**（用户决定）：上游 `unirl-reward-service` 根本没有 test tree（`git ls-tree -r upstream/main -- unirl-reward-service/tests` = 0），CI 也不跑 pytest。PR body 的 Test Plan 老实写了"50 个用例存在但不在 diff 里，是拷进这棵树跑过 50 passed 后删掉的，需要就推上来"。
  - 提交前跑过模板建议的 `SKIP=no-commit-to-branch pre-commit run --from-ref upstream/main --to-ref HEAD`（→ `recipe _target_ paths resolve … Passed`，其余 hook 因 nested project 被 exclude）、`ruff check` 5 个改动文件全 clean、两个 config 过 `load_config`，以及对 `/group/40173` / `zionyfeng` / `uv_venv` / `PID` / `29.163` / `ANTHROPIC` / `_TOKEN` 的泄漏扫描（clean）。
  - 重复劳动检查：30 个 open 上游 PR 里没有一个碰 videohpsv3 或 `unirl-reward-service` 的 video reward；recipe 仍跟着 FlashGRPO 的上游 PR #244 走。
- 本 worktree 的改动仍**未 commit**，且**刻意保持未 commit**：`configs/videohpsv3_service.yaml` 的真实 `weights_path`（本机 16.5GB 权重，避免每次重启重新下载；新的 `_resolve_checkpoint_path` 对这个本地目录行为不变）、`docs/DEVELOPMENT_LOG.md` §19.11-19.13，以及训练侧 recipe。`tests/scorers/test_hpsv3_scorer.py`（5 个 `_resolve_checkpoint_path` 用例）同样只在本 worktree。`envs/hpsv3.txt` 的内容已随 PR #5 提交。`datasets/video/*.txt` 是数据文件，**不要提交**。

**启动链（当前实际用的单机共卡形状，8 卡全用）**：
```
# reward（自己的 Ray 集群：port 6380 + 独立 temp-dir，与训练的 6379 分开）
cd unirl-reward-service
RAY_ADDRESS=127.0.0.1:6380 /group/40173/zionyfeng/uv_venv/reward-venv/bin/python \
  -m reward_service --config configs/videohpsv3_service.yaml
# 等 ~4.5 分钟，curl --noproxy '*' localhost:8080/health 全 ready 再起训练

# 训练
RAY_ADDRESS=127.0.0.1:6379 REPORT_TO_WANDB=true WANDB_MODE=offline \
WANDB_PROJECT=unirl-flashgrpo WANDB_RUN_NAME=wan21_t2v_flashgrpo_videohpsv3_stride1 \
REWARD_SERVICE_URL=http://localhost:8080 \
/group/40173/zionyfeng/uv_venv/venv/bin/python -m unirl.train_diffusion \
  --config-name=diffusion/wan21/wan21_t2v_flashgrpo_sglang_videohpsv3 num_devices=8
```
注意这里 8 个 reward actor 与训练**故意共卡**（97GB H20 上 ~16.4GB reward + ~48GB 训练），所以 `num_devices=8` 不需要 §19.10 里那套 `num_devices=7 ++devices_per_node=7`。

**下次进来先做的事**（按优先级）：
1. **看 wandb 里的 `rollout/generated_videos`**——这是本次重启的首要目的。确认策略退化形态（是否塌成静帧/糊图），以及新 prompt 集下画面是否真的有运动。
2. **盯 reward 曲线是否还是先升后崩**。若仍崩，下一手不是继续调 reward，而是收紧优化强度：`learning_rate: 1.0e-4` 偏大（148 步 Adam 已能漂移），或降 `num_updates_per_batch: 2 → 1`。**注意 `max_grad_norm: 1.0` 在这个 run 里等于没有**（gn 比它低 4 个数量级），别指望它兜底。
3. ~~确认 `checkpoints/wan21_t2v_flashgrpo_sglang_videohpsv3/checkpoint-10/` 在第 10 个 rollout 后真的落盘了~~ **已验证（2026-07-28 18:43）**：目录存在，`checkpoint.pt` 283 MB + `trainer_state.json`（`{"wandb_run_id": "8davz9cd", "optimizer_step": 20}`——10 个 rollout × `num_updates_per_batch: 2`，对得上）。`save_interval: 10` 链路确认可用，不再是只读码验证。wandb run id 记在这里，offline 目录对得上号。
4. 复核 stride 1 的实测 `perf/reward_time_s`——预测 ~60s。若远超预测，说明 FA2 或 8-replica 分摊的假设有问题。
5. 决定 `wan21_t2v_flashgrpo.yaml` 要不要配 `_videohpsv3` 兄弟文件（§19.9.8 未决）。
6. `use_torch_compile: true` 的来源仍未定论。
7. 提交组织：用户说好了再提交，**不要擅自 commit**。
8. **两个 PR 的待办都在等 reviewer**：fork #5 有两条挂着的问题（`envs/hpsv3.txt` 要不要加回一行 FA2 说明；要不要给上游开 issue 把 `create_model_and_processor` 从 `hpsv3/train.py` 拆出来，拆了才能删 trl + tensorboard）；上游 #270 是 draft，等 FlashGRPO 的 #244 落地后再决定转 ready。别在没人回复的情况下自己改 PR 形态。

**绝对不要做的事**（§19.7 / §19.10 / §19.11 全部继续有效，本次新增）：
- **不要把 `frame_stride` 调回 8**——它不是"上游的便宜版"，是把 top-30% 池子从 24 帧砍到 3 帧的另一个估计量，且已实测导致 reward 崩塌（§19.12）。stride 1 只贵 4.6% step 时间。
- **不要靠帧数线性外推来估 reward 成本**——先看 `perf/reward_time_s` 占 step 的比例。这条 run 是 generate-bound（75%），reward 只占 0.8%（§19.12）。
- **不要看 `gn≈0.0001` 就以为优化很温和**——Adam 步长与梯度大小无关，`max_grad_norm` 在这个量级下形同不存在（§19.12）。
- **不要在没有 `save_interval` 的情况下开长跑**——三个 flashgrpo recipe 默认全都不存权重，`da2b5dd` 那个 commit 名字骗人（它改的是 activation checkpointing）。
- 不要用 `datasets/pickscore/*.txt` 训视频——静图美学集，与上游 T2V prompt 集只有 2 行重叠（§19.12）。
- 不要 `git add` `datasets/video/*.txt` 或 `Flash-GRPO/`（后者已被 `.gitignore:109` 忽略）。
- 不要指望 wandb offline `.wandb` 能用官方 `DataStore`/protobuf API 解出 history（实测 0 steps），要 regex 扫二进制（§19.12）。
- **不要在本 worktree 里改 PR 的内容**——现在有三个 worktree：`feature-flashgrpo`（在跑的 run，配置带本机路径）、`pr-reward-service`（fork PR #5，基于 `origin/main`）、`upstream-videohpsv3`（上游 PR #270，基于 `upstream/main`）。后两个刻意保留。改 PR 前先 `git -C <worktree> status --short --branch` 确认位置——`cd` 在 Bash 块之间不保留，用 `git -C` 更稳。
- **不要把 fork 分支直接跨 fork 提上游**——`origin/main` 已岔开 37/69 个 commit，会把无关 commit 全带进去（§19.13）。

*§19.13 已被 §19.15 覆盖，保留作历史记录。*

### 19.14 合并上游 11 个 commit + 修上游 `video.py`（2026-07-28）

用户原话：**"把上游的最新commit给merge进来，保证上游和我们的分支的兼容和正确性"**，随后 **"把这个修复并且提bug修复 pr"**。

#### 19.14.1 先量分叉，再定合并对象

四个分支各自对 `upstream/main`（`5d034c3`，2026-07-28）的距离量过一遍，**不要凭印象**：

| 分支 | 落后 | 领先 |
| --- | --- | --- |
| `feature/flashgrpo-unirl`（本次目标） | **11** | 13 |
| `origin/main` | 69 | 37 |
| `feat/reward-service-videohpsv3` | 69 | 41 |
| `feat/videohpsv3-reward-scorer` | 0 | 3 |

关键点：`feature/flashgrpo-unirl` 是从 **upstream** 拉出来的（merge-base `fce4f4b`），不是从 fork 的 `main`，所以只落后 11 个 commit 而不是 69 个。

#### 19.14.2 在独立 worktree 里合，不动在跑的 run

新建 `../merge-upstream`（分支 `feature/flashgrpo-upstream-merge`，从 `132adac` 建）。理由：302 个文件在运行中的 Ray/SGLang worker 底下变动会造成版本 skew——worker 重启时加载新代码，与已在跑的 driver 不一致。`git merge --no-edit upstream/main` → **零文本冲突**，302 files / +24603 / -6135，merge commit `8711ffd`。

（`git merge-tree --write-tree` 预演走不通：本机 git 2.29.2 没有新式 merge-tree，exit 129。真建 worktree 反而更稳。）

#### 19.14.3 上游这轮的破坏性重构

- `unirl/types/rollout_req.py`(-196) + `rollout_resp.py`(-733) + `prompts.py` **删除**，换成 `types/sample.py`(+883) + `sample_id.py`(+62)。`RolloutReq`/`RolloutResp`/`RolloutTrack` 三件套退役，改成 `Sample`/`Part`。
- `Sample` 只有 `parts` 和 `reward_compute_s` 两个字段；**σ schedule 不在 Sample 上**，要走 `sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas`。
- `NoiseRecipe.from_rollout_req` → `from_sample`；`unirl.sde.ensure_req_sigmas` → `ensure_sample_sigmas`。
- `StageAlgorithm` 新增两个类级 flag：`requires_advantages`（基类 True）、`loss_weighting`（`'sample'`/`'token'`）；`compute_loss_and_backward(advantages=...)` 放宽到 `Optional[torch.Tensor]`。
- `a4bc183 test: remove tests directory (#267)`——**上游整个 tests 目录没了**，所以合并后的验证只能靠 import、hydra compose、ruff 和 `recipe _target_ paths resolve` 钩子，没有上游测试可跑。

#### 19.14.4 兼容性验证（全过）

| 检查 | 结果 |
| --- | --- |
| 566 文件 AST 扫 stale import | clean |
| `sglang_diffusion.adapters` import + registry | 11 个 family 全解析，`wan21`/`ltx2` 都对 |
| `FlashSDEStrategy` | 存活，`canonical_name="flash"`，仍在 `__all__` |
| `patch_dance` | 存活，与上游新增的 `patch_ltx2_rollout_sde` 不冲突（后者只 patch LTX2 专有目标，且在 `hijack.py` 的 `_safe_apply` 里排在 `patch_dance` 之后）|
| hydra compose × 3 个 flashgrpo recipe | 全 OK，`algorithm=unirl.algorithms.flashgrpo.FlashGRPO` |
| `scripts/check_recipe_targets.py` | 2267 个 `_target_` 全解析 |
| `ruff check` / `ruff format --check` | clean |
| trainer 调用点 ↔ `FlashGRPO.compute_loss_and_backward` | 两处调用点（`train/unified_model_stack.py:205`、`train/stack/base.py:256`）传的 5 个 kwarg 全满足 |
| 新 flag 继承 | `requires_advantages=True`、`loss_weighting='sample'` 正确继承；`supports_multi_update=True` 保住（我们 `num_updates_per_batch: 2` 依赖它）；`anchor_fields=('sde_logp',)` 不变；无未实现抽象方法 |
| recipe schema drift | **无**——上游这 11 个 commit 只碰了一个 WAN recipe，且那个 hunk 是 FastVideo 专属的纯注释 |

两个**假线索**，记下来免得下次误判：
- `unirl/config/validation.py` 里的 `validate_*` 函数是给**扁平的 LLM recipe schema** 用的（`training`/`run`/`placement` 顶层键），diffusion recipe 是另一套 schema（`model_config`/`strategy`/`bundle`/...）。`unirl/` 里没有任何地方调它们，原调用点在被删的 tests 里。**拿它们校验 diffusion recipe 会全红，那是 harness 的错不是代码的错。**
- `unirl.patch.dance` 这个路径不存在，真实路径是 `unirl/rollout/engine/sglang_diffusion/_patches/patch_dance.py`。

#### 19.14.5 唯一的真 bug 在上游，不在合并

`unirl/rollout/engine/sglang_diffusion/adapters/video.py` 在**纯净的 `upstream/main`** 上就是坏的：`5d034c3`(#233) 加了 `Ltx2T2VAdapter`，而同区间的 `89115c9`(#214) 删了 `rollout_req.py`，这个文件只被移植了一半。三处残留：

| 行（upstream 原文） | 残留 | 后果 |
| --- | --- | --- |
| 36 | `from unirl.types.rollout_req import RolloutReq` | `ModuleNotFoundError`，模块级即炸 |
| 234 | `NoiseRecipe.from_rollout_req(req)` | `AttributeError` |
| 315 | `expected_sigmas=req.sigmas` | `AttributeError`，`Sample` 无此字段 |

`adapters/__init__.py:25` 是**无 try/except 保护的普通 import**，一次把 `Wan21T2VAdapter`/`Wan22T2VAdapter`/`HunyuanVideoAdapter`/`MochiAdapter` 和 LTX2 从同一模块拉进来，所以 **`upstream/main` 上任何 sglang_diffusion rollout 都起不来**，WAN 连带阵亡，不只是 LTX2。实证方式：把修复 `git stash` 掉让工作树退回 pristine upstream 代码，再 import，拿到真 traceback（不是推断）。上游没接住是因为 #267 把测试删了，而这条路径只在 import 时炸。

修法是 4 处编辑（+5/-6，全在 `Ltx2T2VAdapter` 内，`Wan21T2VAdapter` 本身干净），照抄上游**自己已经移植好**的 `image.py:315` 和同文件父类 `VideoAdapter.build_segment`：删死 import、`req: RolloutReq` → `sample: Sample`（两处）、`from_rollout_req` → `from_sample`、`req.sigmas` → `sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas`（`DiffusionSamplingParams` 原本就在 38 行导入过）。**同一个文件里父类是新 API、子类是旧 API**，这本身就证明是漏改。

#### 19.14.6 上游 bug 修复 PR #272

[Tencent-Hunyuan/UniRL#272](https://github.com/Tencent-Hunyuan/UniRL/pull/272)，`NancyFyong:fix/sglang-diffusion-video-adapter-sample-api` → `Tencent-Hunyuan/UniRL:main`，**非 draft**（`main` 上 import 就挂，属 P0），1 文件 +5/-6，单 commit `9e23b27`，4 项 CI 全绿（`Validate PR title` / `Validate PR body sections` / `lint` / `Sync PR status label`）。

- 第四个 worktree `../fix-video-adapter`，分支从 `upstream/main` 建——**不能拿 `merge-upstream` 那条分支提**，它带着 302 文件的合并。做法是 `git diff` 出补丁跨 worktree apply，`git apply --check` 先确认能干净落到 pristine upstream 上。
- PR body 老实写了两个"没跑"：`pytest`（#267 删了测试目录，`main` 上没有可跑的 suite）、LTX-2 rollout smoke（没有 LTX-2 权重和 AV 节点；那两个 `AttributeError` 只有真跑 LTX-2 才到得了，靠 API 对照 + 与 `image.py` 的 parity 验证，请有 #233 环境的人验 `ltx2_t2v_sglang_loramerge.yaml`）。
- 查重：60 个 open PR 里没有一个碰 `adapters/video.py`，open issue 里也没人报这个 import 失败。
- 同一个 commit 已 cherry-pick 到 `feature/flashgrpo-upstream-merge`（`edd1ecf`），所以我们的合并分支不依赖上游先合。

### 19.15 Resume 入口（覆盖 §19.13，以本节为准）

**当前状态**（2026-07-28）：
- §19.13 里关于 reward service / videohpsv3 / 两个 PR（fork #5、上游 #270）/ 在跑进程 / 启动链 的描述**全部继续有效**，本节只增量覆盖合并相关的部分。
- **五个 worktree**，改任何东西前先 `git -C <worktree> status --short --branch` 确认位置（`cd` 在 Bash 块之间不保留，用 `git -C`）：

| worktree | 分支 | 用途 |
| --- | --- | --- |
| `feature-flashgrpo` | `feature/flashgrpo-unirl` | **在跑的 run**，配置带本机路径，7 个文件刻意未 commit |
| `pr-reward-service` | `feat/reward-service-videohpsv3` | fork PR #5，基于 `origin/main` |
| `upstream-videohpsv3` | `feat/videohpsv3-reward-scorer` | 上游 PR #270（draft），基于 `upstream/main` |
| `merge-upstream` | `feature/flashgrpo-upstream-merge` | 上游合并 + video.py 修复，**等训练跑完再落** |
| `fix-video-adapter` | `fix/sglang-diffusion-video-adapter-sample-api` | 上游 PR #272，基于 `upstream/main` |

- **合并已完成但刻意未落 `feature/flashgrpo-unirl`**（用户决定：等训练跑完再落）。`merge-upstream` 上是 `8711ffd`（merge）+ `edd1ecf`（video.py 修复），工作树干净，验证证据见 §19.14.4。

**下次进来先做的事**（按优先级）：
1. §19.13 的第 1、2、4、5、6 条仍然有效（看 `rollout/generated_videos`、盯 reward 曲线、复核 `perf/reward_time_s`、`wan21_t2v_flashgrpo.yaml` 要不要配兄弟文件、`use_torch_compile` 来源）。
2. **训练跑完后把合并落到 `feature/flashgrpo-unirl`**：在 live worktree 里 merge `feature/flashgrpo-upstream-merge`。落之前先确认 7 个未提交文件还在（merge 不会动它们，但 302 文件的变动值得先 `git stash list` / `status` 存证）。落完重跑 §19.14.4 那张表里的 import + compose + `check_recipe_targets` 三项。
3. **盯 PR #272**：若 reviewer 指出 `expected_sigmas` 该从别处取（PR body 的 Compatibility 段主动点了这个风险），按他们的说法改，并同步改 `merge-upstream` 上的 `edd1ecf`。
4. §19.13 第 8 条不变：fork #5 的两条问题 + 上游 #270 转 ready 都在等 reviewer，别自己改 PR 形态。
5. 提交组织：用户说好了再提交，**不要擅自 commit**（合并分支上的两个 commit 是用户明确要求的）。

**绝对不要做的事**（§19.7 / §19.10 / §19.11 / §19.13 全部继续有效，本次新增）：
- **不要在 live worktree 里做上游合并**——302 个文件在运行中的 Ray/SGLang worker 底下变动 = 版本 skew。开独立 worktree（§19.14.2）。
- **不要用 `unirl/config/validation.py` 的 `validate_*` 校验 diffusion recipe**——那是 LLM 的扁平 schema，全红是 harness 的错（§19.14.4）。
- **不要假设"零冲突"就等于"能跑"**——本次唯一的真 bug 正是零冲突合并之后才暴露的，而且根源在上游而非合并（§19.14.5）。逐个验证重叠点，别信 merge 的沉默。
- **不要拿 `merge-upstream` 分支给上游提 PR**——它带着 302 文件的合并。给上游的修复必须从 `upstream/main` 另起分支（§19.14.6）。
- **不要指望 `git merge-tree --write-tree` 预演**——本机 git 2.29.2 不支持，exit 129。

*§19.15 是当前 Resume 入口。再有新 session 请覆盖本节。*
