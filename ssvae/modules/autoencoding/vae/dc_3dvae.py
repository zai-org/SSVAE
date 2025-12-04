import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from beartype import beartype
from beartype.typing import Union, Tuple
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class CausalConv3d(nn.Module):
    @beartype
    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            pad_mode='constant',
            space_stride=1,
            pad_2d=None,
            **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) #+ (1 - stride)
        if pad_2d is None:
            height_pad = height_kernel_size // 2
            width_pad = width_kernel_size // 2

            self.height_pad = [height_pad, height_pad]
            self.width_pad = [width_pad, width_pad]
        else:
            self.height_pad = [pad_2d[0], pad_2d[1]]
            self.width_pad = [pad_2d[2], pad_2d[3]]


        self.time_pad = time_pad

        stride = (stride, space_stride, space_stride)

        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        if self.pad_mode == 'constant':
            causal_padding_3d = (self.width_pad[0], self.width_pad[1], self.height_pad[0], self.height_pad[1], self.time_pad, 0)
            x = F.pad(x, causal_padding_3d, mode='constant', value=0)
        elif self.pad_mode == 'first':
            if self.time_pad > 0:
                pad_x = torch.cat([x[:, :, :1]]*self.time_pad, dim=2)
                x = torch.cat([pad_x, x], dim=2)
            causal_padding_2d = (self.width_pad[0], self.width_pad[1], self.height_pad[0], self.height_pad[1])
            x = F.pad(x, causal_padding_2d, mode='constant', value=0)
        elif self.pad_mode == 'reflect':
            # reflect padding
            reflect_x = x[:, :, 1:self.time_pad + 1, :, :].flip(dims=[2])
            if reflect_x.shape[2] < self.time_pad:
                reflect_x = torch.cat(
                    [torch.zeros_like(x[:, :, :1, :, :])] * (self.time_pad - reflect_x.shape[2]) + [reflect_x], dim=2)
            x = torch.cat([reflect_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode='constant', value=0)
        else:
            raise ValueError("Invalid pad mode")
        return self.conv(x)


def GroupNormalize3D(in_channels):  # same for 3D and 2D
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = super().forward(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        else:
            x = rearrange(x, "b c h w -> b h w c")
            x = super().forward(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


def LayerNormalize3D(in_channels):
    return LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=False, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


# shuffle3d: spatial-temporal upsample
def pixel_shuffle3d(x, factor):
    B, C, T, H, W = x.size()
    x = x.view(B, -1, factor, factor, factor, T, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(B, -1, T*factor, H*factor, W*factor)
    return x


# shuffle2d: spatial upsample
def pixel_shuffle2d(x, factor):
    B, C, T, H, W = x.size()
    x = x.view(B, -1, factor, factor, T, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.view(B, -1, T, H*factor, W*factor)
    return x


# unshuffle3d: spatial-temporal downsample
def pixel_unshuffle3d(x, factor):
    B, C, D, H, W = x.size()
    x = x.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(B, C * factor ** 3, D // factor, H // factor, W // factor)
    return x


# unshuffle2d: spatial downsample
def pixel_unshuffle2d(x, factor):
    B, C, T, H, W = x.size()
    x = x.view(B, C, T, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.view(B, C * factor ** 2, T, H // factor, W // factor)
    return x


class DCAEChannelDuplicate3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor: int,
            new_shuffle=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor ** 3 % in_channels == 0
        self.repeats = out_channels * factor ** 3 // in_channels
        self.new_shuffle = new_shuffle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.new_shuffle:
            x = x.repeat_interleave(self.repeats, dim=1)
            x = x.view(x.size(0), self.out_channels, self.factor, self.factor, self.factor, x.size(2), x.size(3), x.size(4))
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor, x.size(4) * self.factor,
                    x.size(6) * self.factor)
        else:
            x = pixel_shuffle3d(x, self.factor)
            x = x.repeat(1, self.repeats, 1, 1, 1)
        return x[:, :, self.factor - 1:, :, :]


class DCAEChannelDuplicate2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        new_shuffle=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor ** 2 % in_channels == 0
        self.repeats = out_channels * factor ** 2 // in_channels
        self.new_shuffle = new_shuffle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.new_shuffle:
            x = x.repeat_interleave(self.repeats, dim=1)
            x = x.view(x.size(0), self.out_channels, self.factor, self.factor, x.size(2), x.size(3), x.size(4))
            x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
            x = x.view(x.size(0), self.out_channels, x.size(2), x.size(3) * self.factor,
                    x.size(5) * self.factor)
        else:
            x = pixel_shuffle2d(x, self.factor)
            x = x.repeat(1, self.repeats, 1, 1, 1)
        return x


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = CausalConv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        pad_mode='first')
        self.compress_time = compress_time

    def forward(self, x):
        # x b c t h w
        if self.compress_time:
            if x.shape[2] > 1:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = x_first.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
                x_rest = x_rest.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        else:
            # only interpolate 2D
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

        if self.with_conv:
            t = x.shape[2]
            x = self.conv(x)
        return x


def spatial_interpolate(x):
    T = x.size(2)
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    x = rearrange(x, '(b t) c h w -> b c t h w', t=T)
    return x


def spatial_temporal_interpolate(x):
    x_first = spatial_interpolate(x[:, :, 0:1])
    interploated_x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
    x = torch.cat([x_first, interploated_x[:, :, 2:]], dim=2)
    return x


class ResUpsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, out_channels, compress_time=False, spatial_interpolate=False, temporal_interpolate=False, new_shuffle=False):
        super().__init__()
        self.with_conv = with_conv
        self.spatial_interpolate = spatial_interpolate
        self.temporal_interpolate = temporal_interpolate
        if self.with_conv:
            self.conv = CausalConv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        pad_mode='first')
        self.compress_time = compress_time

        if compress_time:
            self.shortcut = DCAEChannelDuplicate3D(in_channels, out_channels, 2, new_shuffle)
        else:
            self.shortcut = DCAEChannelDuplicate2D(in_channels, out_channels, 2, new_shuffle)

    def forward(self, x):
        # x b c t h w
        sc = self.shortcut(x)
        if self.compress_time:
            if x.shape[2] > 1:
                # for video
                # split first frame
                if self.temporal_interpolate and self.spatial_interpolate:
                    x = spatial_temporal_interpolate(x)
                elif not self.temporal_interpolate and self.spatial_interpolate:
                    x = spatial_interpolate(x)
                    x_first, x_rest = x[:, :, 0:1], x[:, :, 1:]
                    x_rest = x_rest.repeat_interleave(2, dim=2)
                    x = torch.cat([x_first, x_rest], dim=2)
                else:
                    x_first, x_rest = x[:, :, 0], x[:, :, 1:]
                    x_first = x_first.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3) # (B, C, H*2, W*2)
                    x_rest = x_rest.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4) # (B, C, 2*(T-1), 2*H, 2*W)
                    x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2) # (B, C, 2T-1, 2H, 2W)
            else:
                # for image, simply repeat spatial dimension
                if self.spatial_interpolate:
                    x = spatial_interpolate(x)
                else:
                    x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        else:
            # only repeat 2D
            if self.spatial_interpolate:
                x = spatial_interpolate(x)
            else:
                x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

        if self.with_conv:
            x = self.conv(x)
        x = x + sc
        return x


class DownSample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None):
        super().__init__()
        self.with_conv = with_conv
        if out_channels is None:
            out_channels = in_channels
        self.compress_time = compress_time
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if not self.compress_time:
                self.conv = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0)
            else:
                self.conv = CausalConv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        space_stride=2,
                                        pad_2d=[0, 1, 0, 1],
                                        pad_mode='first')

    def forward(self, x):
        if not self.compress_time:
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        else:
            x = self.conv(x)
        return x


class DCAEChannelAverage3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor: int,
            pad_mode='constant'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor ** 3 % out_channels == 0
        self.group_size = in_channels * factor ** 3 // out_channels
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        if self.pad_mode == 'constant':
            pad = (0, 0, 0, 0, self.factor - 1, 0)  # (left, right, top, bottom, front, back)
            x = F.pad(x, pad)
        else:
            pad_x = torch.cat([x[:, :, :1]]*(self.factor-1), dim=2)
            x = torch.cat([pad_x, x], dim=2)
        B, C, D, H, W = x.size()
        x = pixel_unshuffle3d(x, self.factor)
        x = x.view(B, self.out_channels, self.group_size, D // self.factor, H // self.factor, W // self.factor)
        x = x.mean(dim=2)
        return x


class DCAEChannelAverage2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor ** 2 % out_channels == 0
        self.group_size = in_channels * factor ** 2 // out_channels

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        B, C, T, H, W = x.size()
        x = pixel_unshuffle2d(x, self.factor)
        x = x.view(B, self.out_channels, self.group_size, T, H // self.factor, W // self.factor)
        x = x.mean(dim=2)
        return x


class ResDownSample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None, pad_mode='constant'):
        super().__init__()
        self.with_conv = with_conv
        if out_channels is None:
            out_channels = in_channels
        self.compress_time = compress_time
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if not self.compress_time:
                self.conv = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0)
            else:
                self.conv = CausalConv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        space_stride=2,
                                        pad_2d=[0, 1, 0, 1],
                                        pad_mode='first')
        if compress_time:
            self.shortcut = DCAEChannelAverage3D(in_channels, out_channels, 2, pad_mode)
        else:
            self.shortcut = DCAEChannelAverage2D(in_channels, out_channels, 2)

    def forward(self, x):
        sc = self.shortcut(x)
        if not self.compress_time:
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        else:
            x = self.conv(x)
        x = x + sc
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, norm, out_channels=None,
                 dropout, pad_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = norm(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        self.norm2 = norm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        x = x.contiguous()
        return x + identity


def freeze_params(module, freeze_list):
    for name, param in module.named_parameters():
        for freeze_name in freeze_list:
            if name.startswith(freeze_name):
                print(f'Freeze param {name}')
                param.requires_grad = False


def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def forward_checkpoint(module, gradient_checkpoint, h):
    if gradient_checkpoint:
        h = checkpoint(
            create_custom_forward(module),
            h,
            use_reentrant=False
        )
    else:
        h = module(h)
    return h


class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, num_mid_blocks=2,
                 temporal_upsample=[True, True, False],
                 dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, pad_mode='first', temporal_compress_times=4,
                 norm="groupnorm", mid_attn=True, down_block=ResDownSample3D, new_shuffle=False,
                 freeze_entire_encoder=False, gradient_checkpoint=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.gradient_checkpoint = gradient_checkpoint

        if norm == "groupnorm":
            self.norm = GroupNormalize3D
        elif norm == "layernorm":
            self.norm = LayerNorm
        else:
            self.norm = RMS_norm
        print("Using {} norm".format(norm))
        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = CausalConv3d(in_channels, self.ch, kernel_size=3, pad_mode=pad_mode)

        ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        all_ready_downsample = False
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()

            block_in = ch * ch_mult[i_level]
            block_out = ch * ch_mult[i_level + 1]
            if all_ready_downsample:
                block_in = block_out

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                           out_channels=block_out,
                                           dropout=dropout, pad_mode=pad_mode,
                                           norm=self.norm))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                block_out = ch * ch_mult[i_level + 2]
                if temporal_upsample[i_level]:
                    if new_shuffle:
                        down.downsample = down_block(block_in, resamp_with_conv, out_channels=block_out, compress_time=True, pad_mode='first')
                    else:
                        down.downsample = down_block(block_in, resamp_with_conv, out_channels=block_out, compress_time=True, pad_mode='constant')
                else:
                    down.downsample = down_block(block_in, resamp_with_conv, out_channels=block_out, compress_time=False)
                all_ready_downsample = True
            self.down.append(down)

        # middle
        self.mid = nn.ModuleList()
        for i in range(num_mid_blocks):
            self.mid.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_in,
                                         dropout=dropout, pad_mode=pad_mode,
                                         norm=self.norm))
            if i != num_res_blocks - 1 and mid_attn:
                self.mid.append(AttentionBlock(block_in))

        # end
        self.out = nn.Sequential(self.norm(block_in),
                                nn.SiLU(),
                                CausalConv3d(block_in, 2 * z_channels, kernel_size=3, pad_mode=pad_mode))

        if freeze_entire_encoder:
            print('Freeze entire encoder.')
            freeze = []
            for name, module in self.named_modules():
                freeze.append(name)
            freeze_params(self, freeze)

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = forward_checkpoint(self.down[i_level].block[i_block], self.gradient_checkpoint, h)
            if i_level != self.num_resolutions - 1:
                h = forward_checkpoint(self.down[i_level].downsample, self.gradient_checkpoint, h)

        for layer in self.mid:
            h = forward_checkpoint(layer, self.gradient_checkpoint, h)

        # end
        out = self.out(h)
        return out

    def get_last_layer(self):
        return self.out[2].conv.weight


class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, num_mid_blocks=2,
                 dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels,
                 temporal_upsample=[False, True, True],
                 pad_mode='first', temporal_compress_times=4, norm="groupnorm",
                 mid_attn=True, up_block=ResUpsample3D, spatial_interpolate=False, temporal_interpolate=False, 
                 new_shuffle=False, gradient_checkpoint=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.gradient_checkpoint = gradient_checkpoint

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if norm == "groupnorm":
            self.norm = GroupNormalize3D
        elif norm == "layernorm":
            self.norm = LayerNorm
        else:
            self.norm = RMS_norm
        print("Using {} norm".format(norm))

        ch_mult = (ch_mult[-1],) + tuple(ch_mult[::-1])
        block_in = ch * ch_mult[0]
        # z to block_in
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)
        # middle
        self.mid = nn.ModuleList()
        for i in range(num_mid_blocks):
            self.mid.append(ResnetBlock3D(in_channels=block_in,
                                          out_channels=block_in,
                                          dropout=dropout,
                                          pad_mode=pad_mode, norm=self.norm))
            if i != num_mid_blocks - 1 and mid_attn:
                self.mid.append(AttentionBlock(block_in))

        # upsampling
        curr_res = 1
        self.up = nn.ModuleList()
        all_ready_upsample = False
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * ch_mult[i_level]
            block_out = ch * ch_mult[i_level + 1]
            if all_ready_upsample:
                block_in = block_out
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock3D(in_channels=block_in,
                                           out_channels=block_out,
                                           dropout=dropout,
                                           pad_mode=pad_mode, norm=self.norm))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != self.num_resolutions - 1:
                block_out = ch * ch_mult[i_level + 2]
                if temporal_upsample[i_level]:
                    up.upsample = up_block(block_in, resamp_with_conv, out_channels=block_out, compress_time=True, spatial_interpolate=spatial_interpolate, temporal_interpolate=temporal_interpolate, new_shuffle=new_shuffle)
                else:
                    up.upsample = up_block(block_in, resamp_with_conv, out_channels=block_out, compress_time=False, spatial_interpolate=spatial_interpolate, new_shuffle=new_shuffle)
                all_ready_upsample = True
            curr_res = curr_res * 2
            self.up.append(up)

        self.out = nn.Sequential(self.norm(block_in),
                                 nn.SiLU(),
                                 CausalConv3d(block_in, out_ch, kernel_size=3, pad_mode=pad_mode))

    def forward(self, z):
        h = self.conv_in(z)

        for layer in self.mid:
            h = forward_checkpoint(layer, self.gradient_checkpoint, h)

        # upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_checkpoint(self.up[i_level].block[i_block], self.gradient_checkpoint, h)
            if i_level != self.num_resolutions - 1:
                h = forward_checkpoint(self.up[i_level].upsample, self.gradient_checkpoint, h)

        h = self.out(h)
        return h

    def get_last_layer(self):
        return self.out[2].conv.weight

