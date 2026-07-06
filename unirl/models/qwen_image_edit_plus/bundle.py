"""QwenImageEditPlusBundle — thin subclass of :class:`QwenImageBundle`.

The Edit-Plus checkpoint ships the same weight layout as base Qwen-Image
(``transformer/`` + ``vae/`` + ``text_encoder/`` + ``tokenizer/`` +
``scheduler/`` subfolders); only ``transformer/config.json`` differs
(``in_channels=64`` to absorb the source-image latent concat). The base
:meth:`QwenImageBundle.from_config` / ``_from_config_locked`` / meta-init
path / ``fcntl`` serialization all apply unchanged — ``in_channels`` is
read from the checkpoint automatically. This subclass exists so the
Edit-Plus package is self-contained under ``unirl/models/<model_name>/``
per the add-model-bundle skill, and so recipes can wire
``_target_: ...QwenImageEditPlusBundle.from_config``.
"""

from __future__ import annotations

from unirl.models.qwen_image.bundle import QwenImageBundle


class QwenImageEditPlusBundle(QwenImageBundle):
    """Qwen-Image-Edit-Plus bundle: transformer (in_channels=64) + VAE +
    Qwen-VL text encoder + scheduler.

    Inherits :meth:`from_config` / :meth:`_from_config_locked` from
    :class:`QwenImageBundle` unchanged —
    :class:`~unirl.models.qwen_image_edit_plus.config.QwenImageEditPlusPipelineConfig`
    is field-for-field compatible with the base config, and the inherited
    classmethod already constructs ``cls`` (the subclass) instances.
    """


__all__ = ["QwenImageEditPlusBundle"]
