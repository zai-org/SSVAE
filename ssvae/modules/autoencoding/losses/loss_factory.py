from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from matplotlib import colormaps
from matplotlib import pyplot as plt

from ....util import default, instantiate_from_config
from ..lpips.loss.lpips import LPIPS
from ..lpips.model.model import weights_init
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss, calc_gen_loss


class GeneralLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, Dict[str, float]] = None,
        additional_log_keys: Optional[List[str]] = None,
        discriminator_config: Optional[Dict] = None,
        average_nll_loss: bool = False,
        log_logits: bool = False,
        lcr_config = None,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss "
                f"calculation, the LPIPS loss will be applied to each frame "
                f"independently."
            )
        assert disc_loss in ["hinge", "vanilla"]
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.learn_logvar = learn_logvar
        self.average_nll_loss = average_nll_loss

        discriminator_config = default(
            discriminator_config,
            {
                "target": "sgm.modules.autoencoding.lpips.model.model.NLayerDiscriminator",
                "params": {
                    "input_nc": disc_in_channels,
                    "n_layers": disc_num_layers,
                    "use_actnorm": False,
                },
            },
        )

        self.discriminator = instantiate_from_config(discriminator_config).apply(
            weights_init
        )
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = default(regularization_weights, {})
        # SSVAE LCR loss config
        self.lcr_config = lcr_config

        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        self.additional_log_keys = set(default(additional_log_keys, []))
        self.additional_log_keys.update(set(self.regularization_weights.keys()))
        self.log_logits = log_logits

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    @torch.no_grad()
    def log_images(
        self, inputs: torch.Tensor, reconstructions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # calc logits of real/fake
        logits_real = self.discriminator(inputs.contiguous().detach())
        if len(logits_real.shape) < 4:
            # Non patch-discriminator
            return dict()
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
        # -> (b, 1, h, w)

        # parameters for colormapping
        high = max(logits_fake.abs().max(), logits_real.abs().max()).item()
        cmap = colormaps["PiYG"]  # diverging colormap

        def to_colormap(logits: torch.Tensor) -> torch.Tensor:
            """(b, 1, ...) -> (b, 3, ...)"""
            logits = (logits + high) / (2 * high)
            logits_np = cmap(logits.cpu().numpy())[..., :3]  # truncate alpha channel
            # -> (b, 1, ..., 3)
            logits = torch.from_numpy(logits_np).to(logits.device)
            return rearrange(logits, "b 1 ... c -> b c ...")

        logits_real = torch.nn.functional.interpolate(
            logits_real,
            size=inputs.shape[-2:],
            mode="nearest",
            antialias=False,
        )
        logits_fake = torch.nn.functional.interpolate(
            logits_fake,
            size=reconstructions.shape[-2:],
            mode="nearest",
            antialias=False,
        )

        # alpha value of logits for overlay
        alpha_real = torch.abs(logits_real) / high
        alpha_fake = torch.abs(logits_fake) / high
        # -> (b, 1, h, w) in range [0, 0.5]
        # alpha value of lines don't really matter, since the values are the same
        # for both images and logits anyway
        grid_alpha_real = torchvision.utils.make_grid(alpha_real, nrow=4)
        grid_alpha_fake = torchvision.utils.make_grid(alpha_fake, nrow=4)
        grid_alpha = 0.8 * torch.cat((grid_alpha_real, grid_alpha_fake), dim=1)
        # -> (1, h, w)
        # blend logits and images together

        # prepare logits for plotting
        logits_real = to_colormap(logits_real)
        logits_fake = to_colormap(logits_fake)
        # resize logits
        # -> (b, 3, h, w)

        # make some grids
        # add all logits to one plot
        logits_real = torchvision.utils.make_grid(logits_real, nrow=4)
        logits_fake = torchvision.utils.make_grid(logits_fake, nrow=4)
        # I just love how torchvision calls the number of columns `nrow`
        grid_logits = torch.cat((logits_real, logits_fake), dim=1)
        # -> (3, h, w)

        grid_images_real = torchvision.utils.make_grid(0.5 * inputs + 0.5, nrow=4)
        grid_images_fake = torchvision.utils.make_grid(
            0.5 * reconstructions + 0.5, nrow=4
        )
        grid_images = torch.cat((grid_images_real, grid_images_fake), dim=1)
        # -> (3, h, w) in range [0, 1]

        grid_blend = grid_alpha * grid_logits + (1 - grid_alpha) * grid_images

        # Create labeled colorbar
        dpi = 100
        height = 128 / dpi
        width = grid_logits.shape[2] / dpi
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        img = ax.imshow(np.array([[-high, high]]), cmap=cmap)
        plt.colorbar(
            img,
            cax=ax,
            orientation="horizontal",
            fraction=0.9,
            aspect=width / height,
            pad=0.0,
        )
        img.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        # manually convert figure to numpy
        cbar_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        cbar_np = cbar_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cbar = torch.from_numpy(cbar_np.copy()).to(grid_logits.dtype) / 255.0
        cbar = rearrange(cbar, "h w c -> c h w").to(grid_logits.device)

        # Add colorbar to plot
        annotated_grid = torch.cat((grid_logits, cbar), dim=1)
        blended_grid = torch.cat((grid_blend, cbar), dim=1)
        return {
            "vis_logits": 2 * annotated_grid[None, ...] - 1,
            "vis_logits_blended": 2 * blended_grid[None, ...] - 1,
        }

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, aux_loss: torch.Tensor, last_layer: torch.Tensor, max_limit=1e4
    ) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        aux_grads = torch.autograd.grad(aux_loss, last_layer, retain_graph=True)[0]

        if aux_grads is not None:
            aux_weight = torch.norm(nll_grads) / (torch.norm(aux_grads) + 1e-4)
            aux_weight = torch.clamp(aux_weight, 0.0, max_limit).detach()
            return aux_weight
        return None

    def get_rec_loss(self, inputs, rec, average_nll_loss=False, type="l1"):
        # rec_loss.size(): (B, 3, F, H, W)
        if type == "l1":
            rec_loss = torch.abs(inputs.contiguous() - rec.contiguous())
        elif type == "l2":
            rec_loss = (inputs.contiguous() - rec.contiguous()) ** 2
        else:
            raise NotImplementedError

        # p_loss.size(): (1, 1, F, 1, 1)
        if self.perceptual_weight > 0:
            if self.dims > 2:
                t = inputs.shape[2]
                inputs_2d, reconstructions_2d = map(
                    lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                    (inputs, rec),
                )
                p_loss = self.perceptual_loss(
                    inputs_2d.contiguous(), reconstructions_2d.contiguous()
                )
                p_loss = rearrange(p_loss, "(b t) c ... -> b c t ...", t=t)
            else:
                p_loss = self.perceptual_loss(
                    inputs.contiguous(), rec.contiguous()
                )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = 0

        nll_loss, weighted_nll_loss = self.get_nll_loss(rec_loss, average_nll_loss)

        return rec_loss, p_loss, nll_loss

    def get_nll_loss(
        self,
        rec_loss: torch.Tensor,
        average_nll_loss: bool = False,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        # element-wise reconstruction loss
        if average_nll_loss:
            weighted_nll_loss = weighted_nll_loss.mean()
            nll_loss = nll_loss.mean()
        else:
            # summed over B, C, F, H, W
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        return nll_loss, weighted_nll_loss

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        split: str = "train",
        regularization_log: Dict[str, torch.Tensor] = {},
        enc_last_layer=None,
    ) -> Tuple[torch.Tensor, dict]:
        # train the generator part
        if optimizer_idx == 0:
            rec_loss, p_loss, nll_loss = self.get_rec_loss(inputs, reconstructions, self.average_nll_loss)
            # GAN generator loss
            if (global_step >= self.discriminator_iter_start and self.discriminator_weight > 0.) or not self.training:
                logits_fake = self.discriminator(reconstructions.contiguous())
                # For patchgan based discriminator
                if isinstance(logits_fake, tuple):
                    g_loss = 0.
                    for logits_fake_i in logits_fake:
                        g_loss = g_loss + calc_gen_loss(logits_fake_i)
                    g_loss = g_loss / len(logits_fake)
                else:
                    g_loss = calc_gen_loss(logits_fake)
                if self.training:
                    d_weight = self.discriminator_weight * self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0, device=rec_loss.device)
            else:
                d_weight = torch.tensor(0.0, device=rec_loss.device)
                g_loss = torch.tensor(0.0, device=rec_loss.device, requires_grad=True)

            loss = nll_loss + d_weight * self.disc_factor * g_loss
            log = dict()

            # Calculate LCR
            if self.lcr_config is not None:
                assert hasattr(self.lcr_config, "thresh") and hasattr(self.lcr_config, "weight"), "lcr_config must have thresh and weight"
                if not isinstance(self.lcr_config.thresh, float):
                    input_size = inputs.size(-1)
                    ac_thresh = self.lcr_config.thresh[input_size]
                else:
                    ac_thresh = self.lcr_config.thresh
                lcr_loss = torch.relu((ac_thresh - regularization_log["avg_localcorr"]))
                lcr_weight = 0.1 * self.lcr_config.weight * self.calculate_adaptive_weight(nll_loss, lcr_loss, 
                        last_layer=enc_last_layer, max_limit=1e8)
                lcr_loss = lcr_weight * lcr_loss
                loss = loss + lcr_loss
                log[f"{split}/loss/lcr_loss"] = lcr_loss.detach().mean()

            # pop LCR keys
            autocorr_keys = ["image_localcorr", "video_localcorr", "avg_localcorr"]
            for autocorr_key in autocorr_keys:
                if autocorr_key in regularization_log:
                    log[f"{split}/{autocorr_key}"] = regularization_log.pop(autocorr_key)
            element_mean_kl_loss = regularization_log.pop("element_mean_kl_loss", None)
            for k in regularization_log:
                if k in self.regularization_weights:
                    reg_loss = regularization_log[k]
                    regularization_log[k] = reg_loss
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                if k in self.additional_log_keys:
                    if k == "kl_loss" and element_mean_kl_loss is not None:
                        log[f"{split}/{k}"] = element_mean_kl_loss.detach().float().mean()
                    else:
                        log[f"{split}/{k}"] = regularization_log[k].detach().float().mean()

            log_dict = {
                f"{split}/loss/total": loss.clone().detach().mean(),
                f"{split}/loss/nll": nll_loss.detach().mean(),
                f"{split}/loss/rec": rec_loss.detach().mean(),
                f"{split}/loss/percep": p_loss.detach().mean(),
                f"{split}/loss/g": g_loss.detach().mean().to(loss.device),
                f"{split}/scalars/logvar": self.logvar.detach(),
                f"{split}/scalars/d_weight": d_weight.detach(),
            }
            log.update(log_dict)
            return loss, log
        elif optimizer_idx == 1:
            if self.discriminator_weight > 0.:
                # second pass for discriminator update
                if global_step >= self.discriminator_iter_start or not self.training:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
                    if isinstance(logits_real, tuple):
                        d_loss = 0.
                        for logits_real_i, logits_fake_i in zip(logits_real, logits_fake):
                            d_loss = d_loss + self.disc_loss(logits_real_i, logits_fake_i)
                        d_loss = d_loss / len(logits_real)
                    else:
                        d_loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)
                else:
                    d_loss = torch.tensor(0.0, requires_grad=True)
                    logits_real = torch.tensor(0.0)
                    logits_fake = torch.tensor(0.0)

                if isinstance(logits_real, tuple):
                    log = {
                        f"{split}/loss/disc": d_loss.clone().detach().mean(),
                        f"{split}/logits/real_image": logits_real[0].detach().mean(),
                        f"{split}/logits/fake_image": logits_fake[0].detach().mean(),
                        f"{split}/logits/real_video": logits_real[1].detach().mean(),
                        f"{split}/logits/fake_video": logits_fake[1].detach().mean(),
                    }
                else:
                    log = {
                        f"{split}/loss/disc": d_loss.clone().detach().mean(),
                        f"{split}/logits/real": logits_real.detach().mean(),
                        f"{split}/logits/fake": logits_fake.detach().mean(),
                    }
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)
                log = {}
            
            return d_loss, log

