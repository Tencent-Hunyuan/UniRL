# UniRL 开发日志（主仓库 `unirl/` 侧）

本文件记录主仓库训练/算法侧的迭代进展，供任何新 session 直接 resume，不依赖对话历史。
（`unirl-reward-service/` 子项目维护独立的 `unirl-reward-service/docs/DEVELOPMENT_LOG.md`，互不混用。）

---

## 2026-07-31 · FlashGRPO 训练崩溃修复（G-2-additive：每样本单 SDE 步）

### 1. 任务目标与关键决策
- **用户原话**：“那按照 g2 的方法做吧，这样是完全符合上游 flashgrpo 的做法吗”。
- **Bug**：`diffusion.resolve_sde_indices(rollout_id)` 每个 rollout 只调用一次 → 一个 rollout 内所有轨迹共享同一个随机 SDE（σ）步 → 每个优化器步只看到一个 σ → 梯度 ≈ 0，训练崩溃。
- **决策（G-2-additive，最小改动）**：为每条 prompt 独立 i.i.d. 抽一个单 SDE 步（从 scheduler 候选池抽，seed=`rollout_id`）；把 rollout 按步分组、每组一次 `generate`；在段上盖一个**新增可选**字段 `sde_index_per_sample [N]`；合并各组后在 DP scatter 前做一次全局 `randperm` shuffle。FlashGRPO 读该字段时每样本 `S==1`，`old_logp = sde_logp[:, :1]`（恒等），绕过共享 `gather_sde_field`/searchsorted。
- **是否符合上游**：保留 FlashGRPO “稀疏 SDE” 性质（每条轨迹恰好 1 个随机步，其余确定性 ODE），额外成本 ≈ 0（无 N× replay）；scatter 前 shuffle 对齐上游 randperm-before-reshape 语义。

### 2. 不变式（务必保持）
1. **非 flash 路径逐字节不变**（FlowGRPO/FlowDPPO/bagel，共享 `sde_indices`）——所有新逻辑 gated 在 `self._flash_per_sample_sde`（trainer）/ `segment.sde_index_per_sample is not None`（algorithm）。
2. `RolloutReq/Resp.concat` 对单元素短路（`len==1` 原样返回）——非 flash 单 job 因此不变。
3. 共享 `sde_indices` + `gather_sde_field` 路径不动。
4. resume 确定性：抽步（`np.random.default_rng(rollout_id)`）与 shuffle（`torch.Generator().manual_seed(rollout_id)`）都 seed 在 `rollout_id`。

### 3. 文件改动
- **修改**
  - `unirl/types/segments/latent.py`：新增 `sde_index_per_sample` 字段（`FieldKind.CONCAT`，默认 `None`）。
  - `unirl/utils/scheduler_utils.py`：`AllSDEScheduler.sde_candidate_pool()`。
  - `unirl/trainer/diffusion.py`：3 个 helper（`_flash_candidate_pool` / `_build_flash_generate_jobs` / `_stamp_sde_index_per_sample`）、`train_step` 改收 `generate_jobs`、分组 generate + 合并、gated pre-scatter shuffle、`train()` 分支、`_build_req` 加 `sde_index_override`。**（后续补丁，同日）** `_flash_candidate_pool` 加 uniform-K fail-fast（`max(pool) < num_inference_steps-1`）；新增 `_check_flash_rectification_pool`，在 `_build_train_side` 里 gated 校验 `rectification_indices == 候选池`。
  - `unirl/algorithms/flashgrpo.py`：`requires_per_sample_sde_index=True`；per-sample `old_logp` / `_rectification_weights_per_sample`；`beta>0` 在 per-sample 路径报 `NotImplementedError`。
  - `unirl/models/wan21/diffusion.py`：`_replay_per_sample`（每样本 `sigma [N]` gather + 固定槽 `[:,0]/[:,1]` 读取，单次 batched forward）。
- **新增测试**（`tests/` 镜像源码路径）
  - `tests/utils/test_scheduler_utils.py`（5）
  - `tests/types/segments/test_latent_sde_index.py`（5）
  - `tests/algorithms/test_flashgrpo_per_sample.py`（7）
  - `tests/trainer/test_diffusion_flash_jobs.py`（13：9 + 4 守卫用例）
  - `tests/models/wan21/test_diffusion_replay_per_sample.py`（6）

### 4. 测试状态
- **36 个新单测全绿**（venv：`/group/40173/zionyfeng/uv_venv/venv`），`ruff check` 干净。
- 触及目录内既有测试无回归（`tests/utils tests/types tests/algorithms tests/trainer` 合计 42 passed = 30 新 + 12 旧）。
- 命令：
  ```bash
  /group/40173/zionyfeng/uv_venv/venv/bin/python -m pytest \
    tests/utils/test_scheduler_utils.py tests/types/segments/test_latent_sde_index.py \
    tests/algorithms/test_flashgrpo_per_sample.py tests/trainer/test_diffusion_flash_jobs.py \
    tests/models/wan21/test_diffusion_replay_per_sample.py -q
  ```

### 5. simplify / review 结论
- **必改**：0 个（review 确认 0 正确性缺陷、4 条不变式全部成立）。唯一硬性缺口——`_replay_per_sample` 缺单测（违反 §4.5）——**已在本次补上**。
- **建议改**
  1. **【已实现 · 2026-07-31】** 候选池若含末步 `T-1` → mixed-K `concat` 崩溃（当前 recipe 池 `[0,10)`、`T=20`，安全）。已在 `_flash_candidate_pool` 加 uniform-K fail-fast（`max(pool) >= num_inference_steps-1` 即配置期大声报错）。
  2. **【已实现 · 2026-07-31】** `rectification_indices`（算法配置）与 scheduler 候选池须一致，两处独立配置、原无交叉校验，不一致会**静默**误缩放 loss（≈ 意外改 LR）。已加 `_check_flash_rectification_pool`，`_build_train_side` gated 校验（`None` 或不等即报错）。
  3. **（待用户裁决 · 触及既有 shared 路径）** `_rectification_weights` 与 `_rectification_weights_per_sample` 的 4 行归一化逻辑重复；可抽 `_normalize_by_pool(...)`，需谨慎。
  4. **（待用户裁决 · 纯改名）** `_flash_per_sample_sde` 命名与同类兄弟 `_uses_ema` 不一致；建议改 `_uses_flash_per_sample_sde`（多处调用点）。
- **可考虑**：`old_logp = sde_logp[:, :1]` 无 `None`/shape 守卫（`eta=0` → 隐晦 TypeError）；固定槽读取加断言/注释（部分已被新测锁定）；`sigma_max = sigmas[1]` 0-dim tensor vs kernel 的 `float()`（数值等价，仅可读性）；抽 `_draw_flash_steps` 让测试直接断言抽样；局部 `per_sample` → 谓词式命名；补 merged-track 兄弟连续性 / ratio≈1 恒等测试。

### 6. 踩坑与未完成
- #1、#2 两个防静默失败守卫**已落地并补测**（4 用例）；剩 #3（DRY 抽取）、#4（改名）仍待用户裁决——均为打磨项，按 surgical 原则未擅自应用。
- 全部改动仍在工作区，**未提交**（按用户要求）。videohpsv3 / `unirl-reward-service/**` / `.gitignore` 的改动属于另一任务，勿混入本次 FlashGRPO 提交。

---

## Resume 入口（最新状态 · 2026-07-31）

**快照**：G-2-additive 修复（5 源文件）+ 2 个防静默失败守卫（mixed-K fail-fast + rectification/池一致性校验）已落地；5 测试文件 **36 用例全绿**；ruff 干净；未提交。三个 flashgrpo recipe 均满足新守卫（池 `[0,9]` < `T-1=19`、`rectification_indices==[0..9]`）。

**训练已用修复代码重启（2026-07-31 20:08）**：旧崩溃进程 PID 3766411 已停（跑满 2d7h、reward 崩溃、checkpoint-10~50 全为崩溃权重，已归档至 `checkpoints/wan21_t2v_flashgrpo_sglang_videohpsv3.collapsed-run1/`）。新进程 **PID 3620475**（PPID=1，detach），recipe 不变（`wan21_t2v_flashgrpo_sglang_videohpsv3`，num_devices=8），**从头训练**（无 `load_dir`），复用常驻训练 Ray head（:6379），reward 集群（:6380 / reward_service :8080）未动。日志 `logs/train_videohpsv3_bs24_fix.log`（旧日志 `train_videohpsv3_bs24.log` 保留）。启动用最小 env（`RAY_ADDRESS`/`REWARD_SERVICE_URL`/`WANDB_*`），worker 由 raylet 派生继承 fabric env——**未落任何秘密到磁盘**（environ 快照被安全策略拦下，改走最小 env）。

**下次进来先做（优先级）**：
1. 观察新 run 是否还崩溃：看 `logs/train_videohpsv3_bs24_fix.log` 的 reward 曲线 / 新 `checkpoints/wan21_t2v_flashgrpo_sglang_videohpsv3/` 是否在写；确认每优化器步跨多个 σ（修复生效）。
2. 等用户对 §5「建议改」剩余 #3（抽 `_normalize_by_pool`，触及 shared 路径）、#4（`_flash_per_sample_sde`→`_uses_flash_per_sample_sde` 改名）的裁决。二者均为打磨项，非必需。
3. 仅当用户要求提交时再 commit。

**绝对不要做**：
- 未经用户明确指令**停新训练进程 PID 3620475**（或其后继）。
- 误停 reward 集群（reward-venv / :6380 / reward_service :8080 / `ScorerActor`）或常驻训练 Ray head（:6379, gcs 413809 / raylet 420895）。
- 改非 flash 路径语义 / 碰共享 `sde_indices`、`gather_sde_field`。
- `git add .` / `git add -A` / `git add .gitignore`；`/Flash-GRPO/` hunk 保持未提交。
- 把 videohpsv3 相关改动与本次 FlashGRPO 改动混在一个提交里。
