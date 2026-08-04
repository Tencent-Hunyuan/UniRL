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

`AgenticRolloutEngine` is the production coordinator. Rank 0 expands `P` prompts into `P * n`
single-trajectory tasks. Each worker runs up to `per_worker_concurrency` synchronous trajectories
in a thread pool; concurrent calls let the inner backend batch generation while other threads wait
on tools. `generate` is a barrier call. `submit`, `poll`, and `abort` provide the background
partial-rollout interface.

## Trajectories and observations

A trajectory is one ordered `Sample.parts` chain:

```text
input(user) -> generated(assistant) -> observation(tool) -> generated(assistant) -> ...
```

`Sample.fork` appends a generated `Part` carrying sampling parameters and lineage-derived sample
IDs. The engine fills that frontier. `Sample.observe` appends a branch-one input `Part`, defaults
its role to `tool`, and carries no sampling parameters, so observations are conditioning rather
than trainable policy output. The next generation reconstructs its chat history from the ancestor
chain. The production engine attaches an environment-supplied `info["reward"]` to the final
generated `Part`; tool-only environments normally omit it.

### Partial rollout and resume

On `abort`, workers stop pulling new work and let each in-flight turn finish. The trajectory is
checkpointed before the next turn, so generation is never interrupted mid-turn. The returned
carried set contains those checkpointed trajectories plus queued tasks that never started. A
trajectory that becomes terminal during quiescence enters the completed buffer instead; call
`poll()` once after `abort()` to collect it.

Resumption starts at `len(sample.gen_parts())`, preserving the same lineage and continuing under
the newly synchronized weights. This works only when `Environment.reset(partial)` preserves the
existing chain and the environment can recover any required state. Stateless `ToolEnvironment`
does this. `StatefulTool` sessions and `AlfworldEnv` episodes are closed at checkpoint teardown and
their reset paths start fresh state, so their current recipes must drop the interrupted tail rather
than carry it unless the environment supplies its own resumable state. Full stateful resume is
deferred: it also requires keeping resource ownership on the same worker (or serializing it), plus
an explicit release path for abandoned tails; disabling checkpoint teardown alone is not sufficient.

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

`AlfworldEnv` owns one simulator episode per trajectory, carries its episode ID in the root
`Part.control`, returns observations and admissible actions, and emits the cumulative environment
return through `info["reward"]`. GRPO siblings select the same game but run separate episodes.

## Errors and limits

- A missing or malformed tool call is treated as a final answer. Unknown tools, invalid arguments,
  and tool exceptions are returned to the model as error text rather than raised.
- `ToolEnvironment.step` requires a `Texts` frontier. Direct batched use keeps rows aligned with
  empty observations after one row finishes while a sibling continues; production execution uses
  one trajectory per task and avoids that mixed-row case.
- The production engine catches an exception from one trajectory, logs it, returns the trajectory
  accumulated so far as terminal, and continues draining other tasks. Coordinator/pull failures
  still fail the drain. Teardown failures are logged and suppressed.
- `max_turns` is the hard engine bound. When an environment also exposes `max_turns`, the production
  engine requires it to match. `ToolEnvironment` also terminates when no row calls a tool;
  `AlfworldEnv` terminates on simulator completion or `max_steps`.
- `ToolAgentHarness.run` calls `Environment.close` from `finally` on success, failure, AND
  suspension; a teardown error is logged, never raised.

Current runnable configurations live under [`examples/deep_research`](../../../examples/deep_research)
and [`examples/alfworld`](../../../examples/alfworld).
