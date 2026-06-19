"""SP boundary adapter for Flux2Transformer2DModel (MODEL-level boundary; see below)."""

import logging

from unirl.train.backend.veomni.sp.diffusion.ulysses import _assert_seq_divisible, _make_rope_slice_hook, _sp, register

logger = logging.getLogger(__name__)


@register("Flux2Transformer2DModel")
def _wrap_flux2(model, sp_group):
    # flux2 uses the MODEL-level boundary (vs block-level for the others): dual->single
    # blocks + text-strip (hidden[:, num_txt_tokens:]). Because num_txt_tokens ==
    # encoder_hidden_states.shape[1] and img_ids/txt_ids are forward ARGS (not derived
    # from the sliced tensors), we slice both streams at the MODEL input (the strip then
    # removes the LOCAL text, leaving image-local) and gather the image-only output at the
    # MODEL exit. pos_embed is called per stream (img_ids, txt_ids); slice each (cos, sin)
    # on dim 0 so the in-forward cat is the joint sliced RoPE.
    sp = _sp()
    get_parallel_state, slice_input_tensor, gather_outputs = (
        sp.get_parallel_state,
        sp.slice_input_tensor,
        sp.gather_outputs,
    )

    import torch.distributed as dist

    sp_size = dist.get_world_size(sp_group)

    def model_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        if kwargs.get("hidden_states") is not None:
            _assert_seq_divisible(kwargs["hidden_states"].shape[1], sp_size, "image stream")
            kwargs["hidden_states"] = slice_input_tensor(kwargs["hidden_states"], dim=1, group=sp_group)
        if kwargs.get("encoder_hidden_states") is not None:
            _assert_seq_divisible(kwargs["encoder_hidden_states"].shape[1], sp_size, "text/encoder stream")
            kwargs["encoder_hidden_states"] = slice_input_tensor(kwargs["encoder_hidden_states"], dim=1, group=sp_group)
        return args, kwargs

    def model_post(_m, _args, _kwargs, out):
        if not get_parallel_state().ulysses_enabled:
            return out
        sample = out.sample if hasattr(out, "sample") else out[0]
        sample = gather_outputs(sample, gather_dim=1, group=sp_group)
        if hasattr(out, "sample"):
            out.sample = sample
            return out
        return (sample, *out[1:])

    model.register_forward_pre_hook(model_pre, with_kwargs=True)
    model.pos_embed.register_forward_hook(_make_rope_slice_hook(sp_group, dim=0))
    model.register_forward_hook(model_post, with_kwargs=True)
    logger.info("diffusion SP: flux2 boundary hooks installed (model-level slice/gather)")
