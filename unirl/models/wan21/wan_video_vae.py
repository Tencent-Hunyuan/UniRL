"""WanVideoVAE — Wan 2.x Video VAE with training optimizations."""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from torch.nn.modules.utils import _triple

CACHE_T = 2


class Conv3dActGradOnlyFunction(torch.autograd.Function):
    """Conv3d that only computes input gradient, not weight gradient."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(weight)
        ctx.input_info = (input.size(), input.stride(), input.dtype)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            input = torch.empty_strided(
                ctx.input_info[0], ctx.input_info[1], dtype=ctx.input_info[2], device=grad_output.device
            )
            grad_input = torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                None,
                _triple(ctx.stride),
                _triple(ctx.padding),
                _triple(ctx.dilation),
                False,
                [0, 0, 0],
                ctx.groups,
                (True, False, False),
            )[0]
        return grad_input, None, None, None, None, None, None


class Conv2dActGradOnlyFunction(torch.autograd.Function):
    """Conv2d that only computes input gradient."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(weight)
        ctx.input_shape = input.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            from torch.nn.grad import conv2d_input

            grad_input = conv2d_input(
                ctx.input_shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
        return grad_input, None, None, None, None, None, None


class Conv2dActGradOnly(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        return Conv2dActGradOnlyFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution with temporal padding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
        self.time_kernel_size = self.kernel_size[0]

    def forward_pad(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return x

    def forward(self, x, cache_x=None):
        x = self.forward_pad(x, cache_x)
        return super().forward(x)


class CausalConv3dActGradOnly(CausalConv3d):
    """CausalConv3d with act-grad-only optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x, cache_x=None):
        x = self.forward_pad(x, cache_x)
        return Conv3dActGradOnlyFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            causal_conv3d = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
            if hasattr(self, "decoder") and getattr(self, "decoder"):
                setattr(causal_conv3d, "decoder", True)
            self.time_conv = causal_conv3d
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            causal_conv3d = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            if hasattr(self, "decoder") and getattr(self, "decoder"):
                setattr(causal_conv3d, "decoder", True)
            self.time_conv = causal_conv3d
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if hasattr(self, "decoder") and getattr(self, "decoder"):
                x1 = x[:, :, 1:]
                x1 = self.time_conv(x1)
                x1 = x1.reshape(b, 2, c, t - 1, h, w)
                x1 = torch.stack((x1[:, 0], x1[:, 1]), 3)
                x1 = x1.reshape(b, c, (t - 1) * 2, h, w)
                x = torch.cat([x[:, :, 0:1], x1], dim=2)
            elif feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0], x[:, 1]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if hasattr(self, "decoder") and getattr(self, "decoder"):
                x = self.time_conv(x)
            elif feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

        if feat_cache is None or feat_idx is None:
            return x
        return x, feat_cache, feat_idx


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 5:
        return rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    if x.dim() == 4:
        return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    raise ValueError(f"Invalid input shape: {x.shape}")


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 5:
        return rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=patch_size, r=patch_size)
    if x.dim() == 4:
        return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        causal_conv3d_block1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        causal_conv3d_block2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        causal_conv3d_shortcut = CausalConv3d(in_dim, out_dim, 1)
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            causal_conv3d_block1,
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            causal_conv3d_block2,
        )
        self.shortcut = causal_conv3d_shortcut if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if hasattr(self, "decoder") and getattr(self, "decoder"):
                x = layer(x)
            elif check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        if feat_cache is None or feat_idx is None:
            return x + h
        return x + h, feat_cache, feat_idx


class AttentionBlock(nn.Module):
    """Causal self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        use_nested_grad_checkpoint=True,
    ):
        super().__init__()
        self.use_nested_grad_checkpoint = use_nested_grad_checkpoint

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        setattr(self.conv1, "decoder", True)

        res_block_1 = ResidualBlock(dims[0], dims[0], dropout)
        attn_block_2 = AttentionBlock(dims[0])
        res_block_3 = ResidualBlock(dims[0], dims[0], dropout)
        for m in (res_block_1, attn_block_2, res_block_3):
            setattr(m, "decoder", True)
        self.middle = nn.Sequential(res_block_1, attn_block_2, res_block_3)

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                res_block = ResidualBlock(in_dim, out_dim, dropout)
                setattr(res_block, "decoder", True)
                upsamples.append(res_block)
                if scale in attn_scales:
                    attn_block = AttentionBlock(out_dim)
                    setattr(attn_block, "decoder", True)
                    upsamples.append(attn_block)
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                resample = Resample(out_dim, mode=mode)
                setattr(resample, "decoder", True)
                upsamples.append(resample)
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        causal_conv3d = CausalConv3d(out_dim, 3, 3, padding=1)
        setattr(causal_conv3d, "decoder", True)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), causal_conv3d)

        self.unsample_splits = [
            len(self.upsamples) // 15 * 10,
            len(self.upsamples) // 15 * 13,
        ]

    def forward(self, x):
        def custom_forward1(x):
            x = self.conv1(x)
            for layer in self.middle:
                x = layer(x)
            for layer in self.upsamples[: self.unsample_splits[0]]:
                x = layer(x)
            return x

        def custom_forward2(x):
            for layer in self.upsamples[self.unsample_splits[0] : self.unsample_splits[1]]:
                x = layer(x)
            return x

        def custom_forward3(x):
            for layer in self.upsamples[self.unsample_splits[1] :]:
                x = layer(x)
            for layer in self.head:
                x = layer(x)
            return x

        if self.use_nested_grad_checkpoint:
            x = torch.utils.checkpoint.checkpoint(custom_forward1, x, use_reentrant=True)
            x = torch.utils.checkpoint.checkpoint(custom_forward2, x, use_reentrant=True)
            x = torch.utils.checkpoint.checkpoint(custom_forward3, x, use_reentrant=True)
        else:
            x = custom_forward1(x)
            x = custom_forward2(x)
            x = custom_forward3(x)
        return x


def count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


class VideoVAE_(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        use_nested_grad_checkpoint=True,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.use_nested_grad_checkpoint = use_nested_grad_checkpoint

        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            use_nested_grad_checkpoint,
        )

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                    x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                )
            else:
                out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.conv2), z, use_reentrant=False)
        out = torch.utils.checkpoint.checkpoint(create_custom_forward(self.decoder), x, use_reentrant=True)
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _replace_conv_with_act_grad_only(model):
    """Replace Conv layers with act-grad-only versions (post-construction)."""
    for name, module in model.named_modules():
        if isinstance(module, CausalConv3d) and not isinstance(module, CausalConv3dActGradOnly):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else dict(model.named_modules())[parent_name]
            new_module = CausalConv3dActGradOnly.__new__(CausalConv3dActGradOnly)
            nn.Conv3d.__init__(
                new_module,
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                (0, 0, 0),
                module.dilation,
                module.groups,
                module.bias is not None,
            )
            new_module._padding = module._padding
            new_module.padding = module.padding
            new_module.time_kernel_size = module.time_kernel_size
            new_module.weight = module.weight
            new_module.weight.requires_grad = False
            if module.bias is not None:
                new_module.bias = module.bias
                new_module.bias.requires_grad = False
            setattr(parent, child_name, new_module)

        elif isinstance(module, nn.Conv2d) and not isinstance(module, Conv2dActGradOnly):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else dict(model.named_modules())[parent_name]
            new_module = Conv2dActGradOnly(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
            )
            new_module.weight = module.weight
            new_module.weight.requires_grad = False
            if module.bias is not None:
                new_module.bias = module.bias
                new_module.bias.requires_grad = False
            setattr(parent, child_name, new_module)


def _get_task_range(total, world_size, rank):
    """Evenly divide tasks across ranks."""
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total)
    return start, end


class _SimpleConfig:
    """Minimal config object for diffusers pipeline compatibility."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DeterministicLatentDist:
    """Drop-in replacement for diffusers' ``DiagonalGaussianDistribution``."""

    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self, *_args, **_kwargs) -> torch.Tensor:
        return self._latent


class _EncoderOutput:
    """Drop-in replacement for diffusers' ``AutoencoderKLOutput``."""

    def __init__(self, latent: torch.Tensor) -> None:
        self.latent_dist = _DeterministicLatentDist(latent)


class WanVideoVAE(nn.Module):
    """Wan Video VAE with training optimizations."""

    def __init__(self, z_dim=16, use_nested_grad_checkpoint=True, use_act_grad_only_conv=True):
        super().__init__()

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.upsampling_factor = 8
        self.z_dim = z_dim

        self.config = _SimpleConfig(
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=mean,
            latents_std=std,
            z_dim=z_dim,
            scaling_factor=1.0,
        )

        self.model = (
            VideoVAE_(
                z_dim=z_dim,
                use_nested_grad_checkpoint=use_nested_grad_checkpoint,
            )
            .eval()
            .requires_grad_(False)
        )

        if use_act_grad_only_conv:
            _replace_conv_with_act_grad_only(self.model)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype, matching the diffusers ``vae.dtype`` convention."""
        return next(self.parameters()).dtype

    def single_encode(self, video, device):
        video = video.to(device)
        return self.model.encode(video, self.scale)

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, device=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode ``(B, C, T, H, W)`` videos (or a list of ``(C, T, H, W)``) to ``(B, z_dim, T_lat, H_lat, W_lat)``."""
        if device is None and isinstance(videos, torch.Tensor) and videos.dim() == 5:
            target_device = videos.device
            latents = self._encode_batched(videos, target_device, tiled, tile_size, tile_stride)
            return _EncoderOutput(latents)
        if device is None:
            raise ValueError("WanVideoVAE.encode: ``device`` is required for the batched/list calling convention.")
        return self._encode_batched(videos, device, tiled, tile_size, tile_stride)

    def _encode_batched(self, videos, device, tiled, tile_size, tile_stride):
        if isinstance(videos, torch.Tensor) and videos.dim() == 5:
            videos = [videos[i] for i in range(videos.shape[0])]

        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size_px = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride_px = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size_px, tile_stride_px)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_states.append(hidden_state.squeeze(0))
        return torch.stack(hidden_states)

    def decode(self, hidden_states, device=None, tiled=True, sp_group=None, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode latents to video."""
        if device is None:
            device = hidden_states.device
        if tiled:
            if sp_group is not None:
                video = self.tiled_parallel_decode(hidden_states, device, tile_size, tile_stride, sp_group)
            else:
                video = self.tiled_decode(hidden_states, device, tile_size, tile_stride)
        else:
            video = self.single_decode(hidden_states, device)
        return video

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        mask = torch.stack([h, w]).min(dim=0).values
        return rearrange(mask, "H W -> 1 1 1 H W")

    def _make_tile_tasks(self, H, W, size_h, size_w, stride_h, stride_w):
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        return tasks

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        B, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = self._make_tile_tasks(H, W, size_h, size_w, stride_h, stride_w)

        out_T = T * 4 - 3
        weight = torch.zeros(
            (B, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=device,
        )
        values = torch.zeros(
            (B, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=device,
        )

        for h, h_, w, w_ in tasks:
            batch = hidden_states[:, :, :, h:h_, w:w_].to(device)
            batch = self.model.decode(batch, self.scale).to(device)
            mask = self.build_mask(
                batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype, device=device)

            th = h * self.upsampling_factor
            tw = w * self.upsampling_factor
            value_slice = values[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]]
            weight_slice = weight[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]]
            value_slice += batch * mask
            weight_slice += mask.expand_as(weight_slice)

        values = values / weight
        return values.clamp_(-1, 1)

    def tiled_parallel_decode(self, hidden_states, device, tile_size, tile_stride, sp_group):
        """Tiled decode with SP parallelism — each rank decodes a subset of tiles."""
        B, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tile_img_h = size_h * self.upsampling_factor
        tile_img_w = size_w * self.upsampling_factor
        tasks = self._make_tile_tasks(H, W, size_h, size_w, stride_h, stride_w)

        out_T = T * 4 - 3
        weight = torch.zeros(
            (B, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=device,
        )
        values = torch.zeros(
            (B, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=device,
        )

        world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        task_start, task_end = _get_task_range(len(tasks), world_size, sp_rank)
        rank_tasks = tasks[task_start:task_end]

        all_decoded = []
        for h, h_, w, w_ in rank_tasks:
            batch = hidden_states[:, :, :, h:h_, w:w_].to(device)
            batch = self.model.decode(batch, self.scale).to(device)
            padding = (0, tile_img_w - batch.shape[-1], 0, tile_img_h - batch.shape[-2])
            if padding != (0, 0, 0, 0):
                batch = F.pad(batch, padding)
            all_decoded.append(batch)

        local_stack = torch.stack(all_decoded, dim=0)
        gathered = [torch.empty_like(local_stack) for _ in range(world_size)]
        max_tasks = max(
            (task_end - task_start)
            for s, e in [_get_task_range(len(tasks), world_size, r) for r in range(world_size)]
            for task_start, task_end in [(s, e)]
        )
        if local_stack.shape[0] < max_tasks:
            pad_n = max_tasks - local_stack.shape[0]
            local_stack = torch.cat(
                [
                    local_stack,
                    torch.zeros(pad_n, *local_stack.shape[1:], dtype=local_stack.dtype, device=local_stack.device),
                ],
                dim=0,
            )

        gathered = [torch.empty_like(local_stack) for _ in range(world_size)]
        dist.all_gather(gathered, local_stack.contiguous(), group=sp_group)
        all_decoded_global = torch.cat(gathered, dim=0)

        for i, (h, h_, w, w_) in enumerate(tasks):
            if i >= all_decoded_global.shape[0]:
                break
            latent_h = hidden_states[:, :, :, h:h_, w:w_].shape[-2]
            latent_w = hidden_states[:, :, :, h:h_, w:w_].shape[-1]
            img_h = latent_h * self.upsampling_factor
            img_w = latent_w * self.upsampling_factor
            decoded_tile = all_decoded_global[i][:, :, :, :img_h, :img_w]

            mask = self.build_mask(
                decoded_tile,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype, device=device)

            th = h * self.upsampling_factor
            tw = w * self.upsampling_factor
            value_slice = values[:, :, :, th : th + decoded_tile.shape[3], tw : tw + decoded_tile.shape[4]]
            weight_slice = weight[:, :, :, th : th + decoded_tile.shape[3], tw : tw + decoded_tile.shape[4]]
            value_slice += decoded_tile * mask
            weight_slice += mask.expand_as(weight_slice)

        values = values / weight
        return values.clamp_(-1, 1)

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = self._make_tile_tasks(H, W, size_h, size_w, stride_h, stride_w)

        out_T = (T + 3) // 4
        data_device = "cpu"
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        for h, h_, w, w_ in tasks:
            batch = video[:, :, :, h:h_, w:w_].to(device)
            batch = self.model.encode(batch, self.scale).to(data_device)
            mask = self.build_mask(
                batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            th = h // self.upsampling_factor
            tw = w // self.upsampling_factor
            values[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += batch * mask
            weight[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += mask

        return values / weight

    @classmethod
    def load_from_diffusers(cls, pretrained_path, **kwargs):
        """Load WanVideoVAE from HuggingFace diffusers format."""
        vae_dir = pretrained_path
        if os.path.isdir(os.path.join(pretrained_path, "vae")):
            vae_dir = os.path.join(pretrained_path, "vae")

        safetensors_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            hf_sd = load_file(safetensors_path)
        else:
            bin_path = os.path.join(vae_dir, "diffusion_pytorch_model.bin")
            hf_sd = torch.load(bin_path, map_location="cpu")

        vae = cls(**kwargs)
        converted_sd = convert_diffusers_state_dict(hf_sd)
        vae.load_state_dict(converted_sd, strict=True)
        return vae


def convert_diffusers_state_dict(hf_sd: dict) -> OrderedDict:
    """Convert HuggingFace AutoencoderKLWan state dict to WanVideoVAE format."""
    new_sd = OrderedDict()

    resblock_map = {
        "norm1.gamma": "residual.0.gamma",
        "conv1.weight": "residual.2.weight",
        "conv1.bias": "residual.2.bias",
        "norm2.gamma": "residual.3.gamma",
        "conv2.weight": "residual.6.weight",
        "conv2.bias": "residual.6.bias",
        "conv_shortcut.weight": "shortcut.weight",
        "conv_shortcut.bias": "shortcut.bias",
    }

    for hf_key, tensor in hf_sd.items():
        roll_key = _convert_single_key(hf_key, resblock_map)
        new_sd[roll_key] = tensor

    return new_sd


def _convert_single_key(hf_key: str, resblock_map: dict) -> str:
    """Convert a single HF key to ROLL key."""

    if hf_key.startswith("quant_conv."):
        return "model.conv1." + hf_key[len("quant_conv.") :]
    if hf_key.startswith("post_quant_conv."):
        return "model.conv2." + hf_key[len("post_quant_conv.") :]

    if hf_key.startswith("encoder."):
        return "model.encoder." + _convert_encoder_key(hf_key[len("encoder.") :], resblock_map)

    if hf_key.startswith("decoder."):
        return "model.decoder." + _convert_decoder_key(hf_key[len("decoder.") :], resblock_map)

    raise ValueError(f"Unknown HF key: {hf_key}")


def _convert_encoder_key(key: str, rb_map: dict) -> str:
    if key.startswith("conv_in."):
        return "conv1." + key[len("conv_in.") :]

    if key.startswith("norm_out."):
        return "head.0." + key[len("norm_out.") :]

    if key.startswith("conv_out."):
        return "head.2." + key[len("conv_out.") :]

    if key.startswith("down_blocks."):
        rest = key[len("down_blocks.") :]
        dot = rest.index(".")
        block_idx = rest[:dot]
        suffix = rest[dot + 1 :]
        return "downsamples." + block_idx + "." + _convert_resblock_suffix(suffix, rb_map)

    if key.startswith("mid_block."):
        rest = key[len("mid_block.") :]
        if rest.startswith("resnets.0."):
            suffix = rest[len("resnets.0.") :]
            return "middle.0." + _convert_resblock_suffix(suffix, rb_map)
        if rest.startswith("attentions.0."):
            suffix = rest[len("attentions.0.") :]
            return "middle.1." + suffix
        if rest.startswith("resnets.1."):
            suffix = rest[len("resnets.1.") :]
            return "middle.2." + _convert_resblock_suffix(suffix, rb_map)

    raise ValueError(f"Unknown encoder key: {key}")


def _convert_decoder_key(key: str, rb_map: dict) -> str:
    if key.startswith("conv_in."):
        return "conv1." + key[len("conv_in.") :]

    if key.startswith("norm_out."):
        return "head.0." + key[len("norm_out.") :]

    if key.startswith("conv_out."):
        return "head.2." + key[len("conv_out.") :]

    if key.startswith("mid_block."):
        rest = key[len("mid_block.") :]
        if rest.startswith("resnets.0."):
            suffix = rest[len("resnets.0.") :]
            return "middle.0." + _convert_resblock_suffix(suffix, rb_map)
        if rest.startswith("attentions.0."):
            suffix = rest[len("attentions.0.") :]
            return "middle.1." + suffix
        if rest.startswith("resnets.1."):
            suffix = rest[len("resnets.1.") :]
            return "middle.2." + _convert_resblock_suffix(suffix, rb_map)

    if key.startswith("up_blocks."):
        return _convert_upblock_key(key[len("up_blocks.") :], rb_map)

    raise ValueError(f"Unknown decoder key: {key}")


def _convert_upblock_key(key: str, rb_map: dict) -> str:
    """Convert decoder up_blocks flat key to ROLL upsamples flat index."""
    dot = key.index(".")
    block_idx = int(key[:dot])
    rest = key[dot + 1 :]

    base = block_idx * 4

    if rest.startswith("resnets."):
        rest2 = rest[len("resnets.") :]
        dot2 = rest2.index(".")
        res_idx = int(rest2[:dot2])
        suffix = rest2[dot2 + 1 :]
        flat_idx = base + res_idx
        return f"upsamples.{flat_idx}." + _convert_resblock_suffix(suffix, rb_map)

    if rest.startswith("upsamplers.0."):
        suffix = rest[len("upsamplers.0.") :]
        flat_idx = base + 3  # upsampler comes after 3 resnets
        return f"upsamples.{flat_idx}." + suffix

    raise ValueError(f"Unknown up_block key: {key}")


def _convert_resblock_suffix(suffix: str, rb_map: dict) -> str:
    """Convert a ResidualBlock suffix from HF to ROLL format."""
    for hf_pattern, roll_pattern in rb_map.items():
        if suffix == hf_pattern or suffix.startswith(hf_pattern):
            return suffix.replace(hf_pattern, roll_pattern, 1)
    return suffix
