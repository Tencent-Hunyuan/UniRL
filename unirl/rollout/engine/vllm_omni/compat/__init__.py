"""Runtime compat shims for vllm-omni's HunyuanImage 3.0 path.

Currently carries one shim:

- ``tokenizer`` — patches ``PreTrainedTokenizer*.convert_tokens_to_ids``
  to return ``0`` instead of ``None`` for single-token string lookups
  on tokens that aren't in the vocab. Needed for the Base
  ``HunyuanImage-3`` checkpoint, where the ``<img_ratio_36>`` token is
  not present and vllm-omni's HI3 model code does ``ratio_36 + 1``
  unconditionally. Verified empirically that the
  ``HunyuanImage-3.0-Instruct`` checkpoint ships the missing tokens
  natively, so the patch is a no-op when running on Instruct (the
  underlying lookup returns a real id and the wrapper is bypassed).

The patch fires via Stage 0's ``worker_extension_cls`` qualname
``...compat.tokenizer.HI3ARWorkerExtension`` — vllm imports the module
in the AR worker subprocess to resolve the class, and the
module-top ``install()`` call runs as a side effect.
"""
