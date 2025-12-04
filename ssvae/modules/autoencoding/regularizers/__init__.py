from typing import Any, Tuple

import torch

from ....modules.distributions.distributions import \
    DiagonalGaussianDistribution
from .base import AbstractRegularizer
from .calc_local_corr import windowed_localcorr_z


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True, average_kl_loss: bool = False, report_mean_klloss: bool = False, lcr_config=None):
        super().__init__()
        self.sample = sample
        self.average_kl_loss = average_kl_loss
        self.report_mean_klloss = report_mean_klloss
        self.lcr_config = lcr_config

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # z.size(): (B, 2*D, F, H, W)
        log = dict()
        posterior = DiagonalGaussianDistribution(z, average_kl_loss=self.average_kl_loss)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode() # (B, D, F, H, W)
        kl_loss = posterior.kl()
        # Average across batches
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        if self.report_mean_klloss and not self.average_kl_loss:
            c, f, h, w = z.size()[1:]
            element_mean_kl = kl_loss / c / f / h / w
            log["element_mean_kl_loss"] = element_mean_kl
        log["kl_loss"] = kl_loss

        # SSVAE Local Correlation Regularization (LCR)
        if self.lcr_config is not None:
            mode_z = posterior.mode()
            # per-channel normalization
            mean = mode_z.mean([0, 2, 3, 4])[None, :, None, None, None]
            std = mode_z.std([0, 2, 3, 4])[None, :, None, None, None]
            mode_z = (mode_z - mean.detach()) / (std.detach() + 1e-6)
            first_frame_or_image_localcorr, video_localcorr, avg_localcorr = \
            windowed_localcorr_z(mode_z, self.lcr_config["window_size"], 
                weight_type=self.lcr_config.get("weight_type", "average"),
                renorm=self.lcr_config.get("ac_renorm", True))
            log["image_localcorr"] = first_frame_or_image_localcorr
            log["avg_localcorr"] = avg_localcorr
            if video_localcorr is not None:
                log["video_localcorr"] = video_localcorr

        return z, log

