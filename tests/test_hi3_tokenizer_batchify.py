"""CPU regression against the pinned Instruct tokenizer; no model weights are loaded."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
REVISION = "2ec2c78bee7d4b94157341fba86c4c2c7b1858b2"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load_module("hi3_compat", REPO / "unirl/models/hunyuan_image3/compat.py")
repair = compat.repair_hi3_tokenizer_batchify


@pytest.fixture(scope="module")
def upstream():
    path = os.environ.get("HI3_TOKENIZER_PATH") or snapshot_download(
        "tencent/HunyuanImage-3.0-Instruct",
        revision=REVISION,
        allow_patterns=["tokenization_hunyuan_image_3.py", "tokenizer.json", "tokenizer_config.json"],
    )
    path = Path(path)
    module = load_module("hi3_tokenizer_reference", path / "tokenization_hunyuan_image_3.py")
    return path, module


@pytest.fixture
def tokenizer(upstream):
    path, module = upstream
    tokenizer = module.HunyuanImage3TokenizerFast.from_pretrained(path)
    compat.repair_hi3_tokenizer_backend(tokenizer, path)
    assert tokenizer._tokenizer.pre_tokenizer is not None
    return tokenizer


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("cfg_factor", [1, 2, 3])
@pytest.mark.parametrize("mode", ["gen_text", "gen_image", "image_to_text"])
def test_batch_matches_corrected_upstream(tokenizer, upstream, batch_size, cfg_factor, mode):
    path, module = upstream
    source = (path / "tokenization_hunyuan_image_3.py").read_text()
    assert source.count("prompt_list=[[]],") == 1
    reference_namespace = {"__name__": "hi3_corrected_reference"}
    exec(
        compile(source.replace("prompt_list=[[]],", "prompt_list=[[] for _ in message_list],"), str(path), "exec"),
        reference_namespace,
    )
    reference = reference_namespace["HunyuanImage3TokenizerFast"].from_pretrained(path)
    compat.repair_hi3_tokenizer_backend(reference, path)
    prompts = ["A cat.", "Describe the small red boat on the lake in detail.", "Hi"][:batch_size]
    messages = [[dict(role="user", type="text", content=prompt)] for prompt in prompts]
    reference_messages = [[dict(role="user", type="text", content=prompt)] for prompt in prompts]
    for items, namespace in [(messages, vars(module)), (reference_messages, reference_namespace)]:
        for row in items:
            if mode != "gen_text":
                info = namespace["ImageInfo"](
                    image_type="gen_image" if mode == "gen_image" else "vit",
                    image_width=1024,
                    image_height=1024,
                    token_width=2,
                    token_height=2,
                    base_size=1024,
                    ratio_index=0,
                )
                row.insert(
                    len(row) if mode == "gen_image" else 0,
                    dict(
                        role="assistant" if mode == "gen_image" else "user",
                        type="gen_image" if mode == "gen_image" else "cond_vit_image",
                        content=info,
                    ),
                )
    kwargs = dict(
        mode="gen_image" if mode == "gen_image" else "gen_text",
        cfg_factor=cfg_factor,
        bot_task="image" if mode == "gen_image" else "auto",
        sequence_template="instruct",
    )
    if batch_size > 1:
        broken = tokenizer.apply_chat_template(batch_message_list=messages, **kwargs)
        assert broken["output"].tokens.shape[0] == cfg_factor
    assert repair(tokenizer)
    wrapped = tokenizer.apply_general_template
    assert not repair(tokenizer)
    assert tokenizer.apply_general_template is wrapped
    actual = tokenizer.apply_chat_template(batch_message_list=messages, **kwargs)
    expected = reference.apply_chat_template(batch_message_list=reference_messages, **kwargs)
    assert_equal(actual, expected)
    assert actual["output"].tokens.shape[0] == batch_size * cfg_factor
    assert actual["output"].real_pos.shape == (batch_size * cfg_factor, 1)


def test_nonbatch_and_positional_calls(tokenizer):
    messages = [dict(role="user", type="text", content="Hello")]
    original = tokenizer.apply_general_template
    expected = original(messages, add_assistant_prefix=True, uncond_p=1.0)
    assert repair(tokenizer)
    assert_equal(tokenizer.apply_general_template(messages, add_assistant_prefix=True, uncond_p=1.0), expected)
    positional = tokenizer.apply_general_template([messages], None, True, "auto", "auto", None, 0.0, 1, True)
    keyword = tokenizer.apply_general_template([messages], add_assistant_prefix=True, batchify=True)
    assert_equal(positional, keyword)
    with pytest.raises(TypeError):
        tokenizer.apply_general_template(messages, unknown_option=True)
    with pytest.raises(ValueError, match="non-empty list"):
        tokenizer.apply_general_template([], batchify=True)


def test_missing_target():
    assert not repair(object())

    class ChangedTokenizer:
        def apply_general_template(self, messages):
            return messages

        def batch_gen_infer(self):
            raise AssertionError("must not run")

    tokenizer = ChangedTokenizer()
    original = tokenizer.apply_general_template
    assert not repair(tokenizer)
    assert tokenizer.apply_general_template == original
