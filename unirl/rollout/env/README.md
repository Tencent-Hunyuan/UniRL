# Agentic environments and tools

This package defines the environment/tool contracts used by agentic rollout. The turn LOOP
itself lives in [`unirl/rollout/harness/tool_agent.py`](../harness/tool_agent.py)
(`ToolAgentHarness` — worker-side task control flow); the distributed runtime hosting it is
[`AgenticRolloutEngine`](../engine/agentic/engine.py); trajectory storage is described by the
[`Sample`/`Part` types](../../types/README.md).

## Contracts

- `Environment.reset(request) -> Sample` performs per-trajectory setup and may augment or replace
  the request.
- `Environment.step(sample) -> (observation, done, info)` consumes the latest generated action.
  An observation is a decoded `Primitive`; `None` appends no turn. The environment, rather than
  the loop, decides when the episode is done.
- `Environment.close(sample)` is an optional, idempotent teardown hook. It must not raise.
- A `Tool` exposes a JSON function schema and `execute(arguments) -> str`. A `StatefulTool` instead
  implements the session lifecycle `session_start`, `execute_session`, and `session_end`.

`Environment` instances are shared by concurrent trajectory threads on a worker. Implementations
must therefore derive state from the `Sample`, keep it behind a per-trajectory key, or guard it
with locks.

## Synchronous execution

One turn is mechanically:

```text
Sample -> fork -> blocking generate -> environment.step
                                      | done
                                      +------> return Sample
                                      | observation
                                      +------> observe -> next turn
```

`ToolAgentHarness.run` calls `environment.reset` once, forks a one-sample continuation per
turn on the `"policy"` engine, and stops when the environment returns `done=True`, after
`max_turns`, or — at a turn boundary — when the runtime requests a cooperative suspension
(`HarnessContext.suspend_requested`).

`AgenticTrainer` expands `P` prompts into `P * n` tasks, and `RolloutManager`
dispatches each through a single engine slot. Ray actor concurrency lets each
worker run up to `per_worker_inflight` synchronous trajectories while the inner
backend batches requests.
`AgenticRolloutEngine.generate` runs one trajectory; manager `collect` provides
group completion, while `quiesce` provides turn-boundary suspension for cleanup.

## Trajectories and observations

A trajectory is one ordered `Sample.parts` chain:

```text
input(user) -> generated(assistant) -> observation(tool) -> generated(assistant) -> ...
```

`Sample.fork` appends a generated `Part` carrying sampling parameters and lineage-derived sample
IDs. The engine fills that frontier. `Sample.observe` appends a branch-one input `Part`, defaults
its role to `tool`, and carries no sampling parameters, so observations are conditioning rather
than trainable policy output. The next generation reconstructs its chat history from the ancestor
chain. Environments produce observations and termination; the trainer scores
valid terminal answers through its configured `RewardService`.

### Manager quiescence

On manager `quiesce`, dispatch pauses and each in-flight turn finishes. A
nonterminal trajectory is checkpointed before its next turn, so generation is
never interrupted mid-turn. The manager returns retained queued or suspended
tasks according to its root filter. The public `AgenticTrainer` does not resume
these across optimizer steps: normal steps block until every requested group is
complete, and failure cleanup quiesces and discards unfinished work.

## Included environments and tools

`ToolEnvironment` reads the latest `Texts` frontier and parses the last Hermes/Qwen-style
`<tool_call>{...}</tool_call>`. It also accepts a balanced JSON object when a stop string removed
the closing tag. A valid call is dispatched and its raw text result becomes the next `tool`
observation; chat-template rendering adds any tool-response markup. Tool schemas are injected into
the inner engine's `chat_template_kwargs` when that configuration supports them, unless the recipe
already supplied `tools`.

The included tools are:

- `CalculatorTool`: whitelisted arithmetic AST evaluation.
- `SandboxTool`: a per-session persistent Python subprocess with timeout and output limits.
- `SearchTool`: batched Serper or SerpApi search; requires `SERPER_KEY_ID`.
- `VisitTool`: Jina page reading with optional OpenAI-compatible summarization.

## Errors and limits

- A missing or malformed tool call is treated as a final answer. Unknown tools, invalid arguments,
  and tool exceptions are returned to the model as error text rather than raised.
- `ToolEnvironment.step` requires a `Texts` frontier. Direct batched use keeps rows aligned with
  empty observations after one row finishes while a sibling continues; production execution uses
  one trajectory per task and avoids that mixed-row case.
- The harness catches an exception from one trajectory, logs it, and returns the partial trace as
  a `failed` outcome; the trainer excludes failed trajectories from GRPO statistics. A slot RPC
  failure poisons the manager and fails the training step. Teardown failures are logged and
  suppressed.
- `max_turns` is the hard engine bound. When an environment also exposes
  `max_turns`, the production engine requires it to match. `ToolEnvironment`
  also terminates when no row calls a tool.
- `ToolAgentHarness.run` calls `Environment.close` from `finally` on success, failure, AND
  suspension; a teardown error is logged, never raised.

The runnable agentic configuration is
[`examples/deep_research/deep_research_search_judge.yaml`](../../../examples/deep_research/deep_research_search_judge.yaml).
