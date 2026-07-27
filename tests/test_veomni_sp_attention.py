from unirl.train.backend.veomni.sp import ar


def test_select_sp_attention_prefers_newest_available(monkeypatch):
    import transformers.utils as transformers_utils

    monkeypatch.setattr(transformers_utils, "is_flash_attn_4_available", lambda: True)
    monkeypatch.setattr(transformers_utils, "is_flash_attn_3_available", lambda: True)
    monkeypatch.setattr(transformers_utils, "is_flash_attn_2_available", lambda: True)

    assert ar._select_sp_attn_impl() == "veomni_flash_attention_4_with_sp"


def test_select_sp_attention_falls_back_to_fa2(monkeypatch):
    import transformers.utils as transformers_utils

    monkeypatch.setattr(transformers_utils, "is_flash_attn_4_available", lambda: False)
    monkeypatch.setattr(transformers_utils, "is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr(transformers_utils, "is_flash_attn_2_available", lambda: False)

    assert ar._select_sp_attn_impl() == ar.SP_ATTN_IMPL
