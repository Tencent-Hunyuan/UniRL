"""Model package.

Holds the typed pipeline-stage protocols (``models.types``) that pair
with the four-tier type system (``Conditions``, ``Segments``, packed
primitives, ``RolloutReq`` / ``RolloutResp``). Each concrete model
(``sd3``, ``wan21``, ``qwen_image``, …) is a sibling subpackage
implementing those protocols.
"""
