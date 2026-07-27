from types import SimpleNamespace

import torch

from unirl.models.qwen3.ar import Qwen3ARStage
from unirl.models.qwen3.conditions import Qwen3ARConditions
from unirl.types.conditions import TextTokenCondition
from unirl.types.sampling import ARSamplingParams


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length


class _FakeTransformer:
    def __init__(self) -> None:
        self.prepared_lengths = []
        self.cache_lengths = []

    def prepare_inputs_for_generation(
        self,
        input_ids,
        *,
        next_sequence_length,
        past_key_values,
        attention_mask,
        use_cache,
    ):
        selected = input_ids[:, -next_sequence_length:]
        self.prepared_lengths.append(int(selected.shape[1]))
        return {
            "input_ids": selected,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def __call__(self, *, input_ids, past_key_values, attention_mask, use_cache, return_dict):
        previous = 0 if past_key_values is None else past_key_values.length
        cache = _FakeCache(previous + int(input_ids.shape[1]))
        self.cache_lengths.append(cache.length)
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=cache)

    @staticmethod
    def _update_model_kwargs_for_generation(out, model_kwargs):
        updated = dict(model_kwargs)
        updated["past_key_values"] = out.past_key_values
        updated["attention_mask"] = torch.cat(
            [updated["attention_mask"], updated["attention_mask"].new_ones((1, 1))],
            dim=-1,
        )
        return updated


def test_autoregress_only_appends_one_token_to_cached_decode() -> None:
    transformer = _FakeTransformer()
    stage = object.__new__(Qwen3ARStage)
    stage.model = SimpleNamespace(
        transformer=transformer,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=None),
    )
    conditions = Qwen3ARConditions(
        prompt=TextTokenCondition(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
    )

    segment = stage.autoregress(
        conditions,
        sampling_params=ARSamplingParams(
            samples_per_prompt=1,
            temperature=0,
            max_new_tokens=4,
            top_p=1,
            top_k=0,
        ),
    )

    assert transformer.prepared_lengths == [3, 1, 1, 1]
    assert transformer.cache_lengths == [3, 4, 5, 6]
    assert segment.lengths.tolist() == [4]
