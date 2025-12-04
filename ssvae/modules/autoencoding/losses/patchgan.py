import torch.nn as nn
from torch.nn.utils import spectral_norm
from math import log2
from ssvae.modules.autoencoding.vae.dc_3dvae import CausalConv3d
from torch.utils.checkpoint import checkpoint


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


class PatchGAN2DDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, max_spatial_down=8, gradient_checkpoint=False):
        super(PatchGAN2DDiscriminator, self).__init__()
        self.gradient_checkpoint = gradient_checkpoint
        n_layers = int(log2(max_spatial_down))
        self.input_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        nf_mult = 1
        nf_mult_prev = 1
        self.blocks = nn.ModuleList()
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            # nf_mult = min(2 ** n, 8)
            nf_mult = min(2 ** n, 16)
            self.blocks.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.out_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        B, C, F, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B*F, C, H, W)
        x = forward_checkpoint(self.input_layer, self.gradient_checkpoint, x)
        for block in self.blocks:
            x = forward_checkpoint(block, self.gradient_checkpoint, x)
        x = self.out_layer(x)
        C, H, W = x.size()[1:]
        x = x.view(B, F, C, H, W).contiguous().transpose(1, 2)
        return x


class PatchGAN3DDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, max_spatial_down=8, max_temporal_down=4, gradient_checkpoint=False):
        super(PatchGAN3DDiscriminator, self).__init__()
        self.gradient_checkpoint = gradient_checkpoint
        n_layers = int(log2(max_spatial_down))
        self.n_temporal_down_layers = int(log2(max_temporal_down))

        self.conv_in = nn.Sequential(
            CausalConv3d(in_channels, ndf, kernel_size=3, stride=2, 
                space_stride=2, pad_2d=[0, 1, 0, 1], pad_mode='first'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        nf_mult = 1
        nf_mult_prev = 1
        down_blocks = nn.ModuleList()
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 16)
            if n < self.n_temporal_down_layers:
                down_blocks.append(
                    nn.Sequential(
                        CausalConv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, 
                        space_stride=2, pad_2d=[0, 1, 0, 1], pad_mode='first', bias=False),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            else:
                down_blocks.append(
                    nn.Sequential(
                        spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, padding=1, bias=False)),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
        self.down_blocks = down_blocks

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.out = nn.Sequential(
            CausalConv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, pad_mode='first', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            CausalConv3d(ndf * nf_mult, 1, kernel_size=3, stride=1, pad_mode='first')
        )

        self.add_spectral_norm_to_conv3d()

    def add_spectral_norm_to_conv3d(self):
        print('Add spectral norm to all CausalConv3d in PatchGAN3DDiscriminator')
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module.conv = spectral_norm(module.conv)

    def forward(self, x):
        x = self.conv_in(x)
        for i, block in enumerate(self.down_blocks):
            if i+1 < self.n_temporal_down_layers:
                x = forward_checkpoint(block, self.gradient_checkpoint, x)
            elif i+1 == self.n_temporal_down_layers:
                B, C, F, H, W = x.size()
                x = x.transpose(1, 2).contiguous().view(B*F, C, H, W)
                x = forward_checkpoint(block, self.gradient_checkpoint, x)
            else:
                x = forward_checkpoint(block, self.gradient_checkpoint, x)
        C, H, W = x.size()[1:]
        x = x.view(B, F, C, H, W).contiguous().transpose(1, 2)
        x = self.out(x)
        return x


class PatchGANImageVideoDisc(nn.Module):
    def __init__(self, in_channels=3, ndf=64, max_spatial_down=8, max_temporal_down=4, gradient_checkpoint=False):
        super(PatchGANImageVideoDisc, self).__init__()
        self.patchgan_2d = PatchGAN2DDiscriminator(in_channels, ndf, max_spatial_down, gradient_checkpoint)
        self.patchgan_3d = PatchGAN3DDiscriminator(in_channels, ndf, max_spatial_down, max_temporal_down, 
            gradient_checkpoint=gradient_checkpoint)

    def forward(self, x):
        image_level_logits = self.patchgan_2d(x)
        video_level_logits = self.patchgan_3d(x)
        return image_level_logits, video_level_logits

