# Supervised (SFT) track builders

> **Where it fits:** the *build* step of the SFT domain. In: normalized manifest
> records from `unirl/data/sft.py`. Out: a training `Part`.
> Full map: [`../../README.md`](../../README.md).

*Records stay opaque driver-side; media loading, tokenization and VAE encode all
happen here, on the training workers.*

## What it is

`ARSupervisedTrackBuilder` (LLM / VLM / agent), `DiffusionSupervisedTrackBuilder`
and `VideoDiffusionSupervisedTrackBuilder`. Each turns one shard of records into
the `Part` its algorithm consumes, using the pipeline's own stages — the prompt
side is whatever rollout would render, so SFT trains on the token sequence
inference will see.

## Agent chat-stage contract

Agent rows (`{"messages": [...]}`) are dispatched on what the pipeline's chat
stage declares it can render, so a new AR backbone implements only what it
supports. There is no base class and no `isinstance` check:

| Attribute | Required | Meaning |
| --- | --- | --- |
| `embed_messages(conversations, *, tools)` | for agent rows | histories → the pipeline's AR conditions |
| `supports_message_images` | flag, default false | image parts in message content are rendered, not dropped |
| `tokenize_agent_target(record)` | optional | the stage owns the target convention; otherwise the HF chat-template suffix path applies |

## Gotchas

- **A stage that cannot render the WHOLE target must raise, not render part of
  it.** Manifests may carry tool-call targets — `unirl/utils/prepare_sft_agent.py`
  supervises them by design — so a backbone with no tool-call template rejects
  the record instead of quietly tokenizing only its text. `BagelChatTemplateStage`
  has no tool rendering at all; `QwenVLChatTemplateStage` rejects too, because the
  stock Qwen2.5-VL chat template references neither `tools` nor `tool_calls` and
  the generic HF suffix path would drop them byte-invisibly. Silently supervising
  the understood fraction makes the same normalized record mean different things
  on different backbones.
- **The conversation is authoritative for the system prompt.** `embed_messages`
  never injects a stage's configured `system_instruction`; that belongs to the
  legacy `embed` path. Both backbones behave the same way, so one
  manifest/config pair cannot render differently per model.
- **`supports_message_images` is checked, not assumed.** A text-only chat stage
  handed an image-bearing history raises rather than dropping the vision
  placeholders — training without the images would otherwise be undetectable.
- **Agent targets are filtered, never truncated.** Overlong agent targets raise
  at `max_response_length`; overlong prompts raise at the chat stage. Cutting an
  image-bearing prompt desyncs pixel grids from the remaining pad tokens, and
  cutting a text prompt severs the prompt→target seam. Filter during manifest
  preparation instead.
