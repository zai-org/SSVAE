import re
import random
import logging
from packaging import version
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.cuda.amp as amp
import pytorch_lightning as pl

from ..util import default, get_obj_from_str, instantiate_from_config
from ..modules.ema import LitEma
from ..modules.autoencoding.regularizers import AbstractRegularizer
from torch.optim.lr_scheduler import LambdaLR

logpy = logging.getLogger(__name__)


class AbstractAutoencoder(pl.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        if isinstance(ckpt, str):
            ckpt = {
                "target": "sgm.modules.checkpoint.CheckpointEngine",
                "params": {"ckpt_path": ckpt},
            }
        engine = instantiate_from_config(ckpt)
        engine(self)

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logpy.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logpy.info(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        trainable_ae_params: Optional[List[List[str]]] = None,
        ae_optimizer_args: Optional[List[dict]] = None,
        trainable_disc_params: Optional[List[List[str]]] = None,
        disc_optimizer_args: Optional[List[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        scheduler_config = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # pytorch lightning

        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        self.regularization: AbstractRegularizer = instantiate_from_config(
            regularizer_config
        )
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        self.scheduler_config = scheduler_config
        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.additional_decode_keys = set(default(additional_decode_keys, []))

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        self.init_from_ckpt(ckpt)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + [p for p in self.encoder.parameters() if p.requires_grad]
        params = params + [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        mean: torch.Tensor = None,
        istd: torch.Tensor = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        with amp.autocast(dtype=x.dtype):
            z = self.encoder(x, **kwargs)
            if mean is not None and istd is not None:
                mean = mean.to(z.device)
                istd = istd.to(z.device)
                mu, logvar = z.chunk(2, dim=1)
                z = (mu - mean) * istd
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, mean=None, istd=None, **kwargs) -> torch.Tensor:
        with amp.autocast(dtype=z.dtype):
            if mean is not None and istd is not None:
                mean = mean.to(z.device)
                istd = istd.to(z.device)
                z = z / istd + mean
            x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        if optimizer_idx == 0:
            z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        else:
            with torch.no_grad():
                z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False, # True may cause NCCL timeout
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        opts[1]._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
        opts[1]._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx, optimizer_idx=optimizer_idx
            )
            if not torch.isnan(loss):
                # complement with automatic_optimization = True
                self.manual_backward(loss)
            else:
                opt.zero_grad()
                print(f'Skip optimization for nan loss at batch_idx {batch_idx}.')
        opt.step()
        if optimizer_idx == 0:
            lr_scheduler = self.lr_schedulers()
            if len(lr_scheduler) > 0:
                lr_scheduler = lr_scheduler[0]
                lr_scheduler.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            logpy.info(
                f"Number of trainable discriminator parameters: {num_disc_params:,}"
            )
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            dis_learning_rate = self.learning_rate / 10
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, dis_learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)

        if self.scheduler_config is not None:
            self.scheduler = instantiate_from_config(self.scheduler_config)
            self.scheduler.lr_max = self.learning_rate
            print("Setting up LambdaLR scheduler...")
            scheduler_ae = {
                "scheduler": LambdaLR(opt_ae, lr_lambda=lambda epoch: self.scheduler(epoch)/self.learning_rate),
                "interval": "step",
                "frequency": 1,
            }

            scheduler_disc = {
                "scheduler": LambdaLR(opt_disc, lr_lambda=lambda epoch: 1.0),
                "interval": "step",
                "frequency": 1,
            }
            return opts, [scheduler_ae, scheduler_disc]

        return opts

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        if isinstance(x, tuple):
            x = x[0]
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        _, xrec, _ = self(x, **additional_decode_kwargs)
        log["inputs"] = x
        log["reconstructions"] = xrec
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        # self.loss.log_images: log discriminator logits map
        if hasattr(self.loss, "log_images") and self.loss.discriminator_weight > 0. and self.loss.log_logits:
            log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


class AutoencodingEngineSSVAE(AutoencodingEngine):
    def __init__(
        self, *args, encoder_config, decoder_config, loss_config, regularizer_config, optimizer_config = None, 
        lr_g_factor = 1, trainable_ae_params = None, ae_optimizer_args = None, trainable_disc_params = None, 
        disc_optimizer_args = None, disc_start_iter = 0, diff_boost_factor = 3, ckpt_engine = None, ckpt_path = None, 
        additional_decode_keys = None, scheduler_config=None, lmr_config=None, **kwargs):
        super().__init__(*args, encoder_config=encoder_config, decoder_config=decoder_config, loss_config=loss_config, 
            regularizer_config=regularizer_config, optimizer_config=optimizer_config, lr_g_factor=lr_g_factor, 
            trainable_ae_params=trainable_ae_params, ae_optimizer_args=ae_optimizer_args, 
            trainable_disc_params=trainable_disc_params, disc_optimizer_args=disc_optimizer_args, 
            disc_start_iter=disc_start_iter, diff_boost_factor=diff_boost_factor, ckpt_engine=ckpt_engine, 
            ckpt_path=ckpt_path, additional_decode_keys=additional_decode_keys, scheduler_config=scheduler_config, **kwargs)

        # SSVAE Latent Masked Reconstruction (LMR)
        self.use_lmr = (lmr_config is not None)
        if self.use_lmr:
            self.mask_ratios = getattr(lmr_config, "mask_ratios", [0.0, 0.25, 0.5, 0.75])
            self.mask_ratios.sort()
            self.mask_probs = getattr(lmr_config, "mask_probs", {256: [0.7, 0.1, 0.1, 0.1], 512: [0.6, 0.1, 0.15, 0.15]})
            self.block_sizes = getattr(lmr_config, "block_sizes", {256: [1, 1, 1], 512: [2, 2, 2]})
            z_channels = getattr(self.decoder, 'z_channels', None)
            if z_channels is None:
                z_channels = encoder_config['params'].get('z_channels', 48)
            self.mask_token = torch.nn.Parameter(torch.randn(1, z_channels, 1, 1, 1))

    def apply_lmr(self, z):
        B, C, T, H, W = z.shape
        input_size = H * 16
        block_size = self.block_sizes[input_size]
        weights = self.mask_probs[input_size]
        mask_ratio = float(np.random.choice(self.mask_ratios, p=weights))
        if mask_ratio < self.mask_ratios[1]:
            return z

        t_block, h_block, w_block = block_size
        # calculate block number
        t_blocks = T // t_block
        h_blocks = H // h_block
        w_blocks = W // w_block
        total_blocks = t_blocks * h_blocks * w_blocks
        total_latent_pixels = T * H * W
        block_area = t_block * h_block * w_block
        num_mask_blocks = max(1, int((total_latent_pixels * mask_ratio) // block_area))
        mask_block_indices = torch.randperm(total_blocks, device=z.device)[:num_mask_blocks]

        mask = torch.zeros(B, 1, T, H, W, device=z.device, dtype=torch.bool)

        # calculate mask coordinates
        t_idx = mask_block_indices // (h_blocks * w_blocks)
        hw_idx = mask_block_indices % (h_blocks * w_blocks)
        h_idx = hw_idx // w_blocks
        w_idx = hw_idx % w_blocks
        t_offsets = torch.arange(t_block, device=z.device)
        h_offsets = torch.arange(h_block, device=z.device)
        w_offsets = torch.arange(w_block, device=z.device)
        t_offsets, h_offsets, w_offsets = torch.meshgrid(t_offsets, h_offsets, w_offsets, indexing='ij')
        t_coords = t_idx[:, None, None, None] * t_block + t_offsets[None, :, :, :]
        h_coords = h_idx[:, None, None, None] * h_block + h_offsets[None, :, :, :]
        w_coords = w_idx[:, None, None, None] * w_block + w_offsets[None, :, :, :]
        t_coords = t_coords.reshape(-1)
        h_coords = h_coords.reshape(-1)
        w_coords = w_coords.reshape(-1)

        mask[:, :, t_coords, h_coords, w_coords] = True
        mask_token_expanded = self.mask_token.expand(z.shape)
        z = torch.where(mask.expand_as(z), mask_token_expanded, z)
        return z

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)

        if self.use_lmr:
            z = self.apply_lmr(z)

        dec = self.decode(z, **additional_decode_kwargs)

        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)

        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            # SSVAE Local Correlation Regularization (LCR)
            if self.regularization.lcr_config is not None:
                extra_info["enc_last_layer"] = self.encoder.get_last_layer()
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False, # True may cause NCCL timeout
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + [p for p in self.encoder.parameters() if p.requires_grad]
        params = params + [p for p in self.decoder.parameters() if p.requires_grad]
        if self.use_lmr:
            params.append(self.mask_token)
        return params


class VideoAutoencodingEngine(AutoencodingEngineSSVAE):
    def __init__(
        self,
        ckpt_path: Union[None, str] = None,
        ignore_keys: Union[Tuple, list] = (),
        image_video_weights=[1,1],
        disable_sampling=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        s = sum(image_video_weights)
        self.image_video_weights = [w/s for w in image_video_weights]
        seed = random.randint(0, 1000000)
        self.rng = np.random.default_rng(seed=[seed])
        self.disable_sampling = disable_sampling

    def log_videos(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        return self.log_images(batch, additional_log_kwargs, **kwargs)

    def get_input(self, batch: dict) -> torch.Tensor:
        if not self.disable_sampling:
            # do image/video sampling here
            index = self.rng.choice(2, p=self.image_video_weights)
            if index == 0:
                return batch["image_batch"][self.input_key]
            else:
                return batch["video_batch"][self.input_key]
        else:
            return batch[self.input_key]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")


class VideoAutoencoderInferenceWrapper(VideoAutoencodingEngine):
    def __init__(
            self,
            dtype=torch.bfloat16,
            mean=None,
            std=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mean = torch.tensor(mean, dtype=dtype, device="cuda").reshape(1, -1, 1, 1, 1) if mean is not None else None
        self.inverse_std = 1.0 / torch.tensor(std, dtype=dtype, device="cuda").reshape(1, -1, 1, 1, 1) if std is not None else None

    def encode(self, x: torch.Tensor, return_reg_log: bool = False, **kwargs):
        z, _ = super().encode(x, return_reg_log, unregularized=True, mean=self.mean, istd=self.inverse_std)
        if return_reg_log:
            z, reg_log = z
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs):
        recon = super().decode(z, mean=self.mean, istd=self.inverse_std)
        return recon

