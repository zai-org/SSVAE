import io
import re
import os
import sys
import json
import numpy as np
from PIL import Image
from functools import partial
import math
import tarfile
from braceexpand import braceexpand
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import torchvision.transforms as TT
from pytorch_lightning import LightningDataModule

from sat import mpu
import webdataset as wds
from webdataset import ResampledShards, DataPipeline
from webdataset.utils import pytorch_worker_seed
from webdataset.filters import pipelinefilter
from webdataset.tariterators import url_opener, group_by_keys
from webdataset.handlers import reraise_exception, warn_and_continue, ignore_and_continue

import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torch.utils.data import IterableDataset

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, av, _read_from_stream, _align_audio_frames
from torchvision.transforms import Compose
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

from ssvae.util import instantiate_from_config
import torch.distributed as dist
from sat.data_utils.webds import tarfile_samples_with_meta, ConfiguredResampledShards


def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    from torchvision import get_video_backend

    if get_video_backend() != "pyav":
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        try:
            with av.open(filename, metadata_errors="ignore") as container:
                if container.streams.audio:
                    audio_timebase = container.streams.audio[0].time_base
                if container.streams.video:
                    video_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.video[0],
                        {"video": 0},
                    )
                    video_fps = container.streams.video[0].average_rate
                    # guard against potentially corrupted files
                    if video_fps is not None:
                        info["video_fps"] = float(video_fps)

                if container.streams.audio:
                    audio_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.audio[0],
                        {"audio": 0},
                    )
                    info["audio_fps"] = container.streams.audio[0].rate

        except av.AVError:
            # TODO raise a warning?
            pass

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def process_video(video_path, image_size=None, duration=None, num_frames=4, wanted_fps=None, actual_fps=None, inference=False):
    '''
        video_path: str or io.BytesIO
        image_size: .
        duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
        num_frames: wanted num_frames.
        wanted_fps: .
    '''
    if duration is not None and not inference:
        max_seek = duration - num_frames / wanted_fps # the later term is the duration of the sampled num_frames clip
        start = random.uniform(0, max_seek)
    else:
        start = 0

    video = read_video(
        video_path,
        start_pts=start,
        end_pts=start + num_frames / wanted_fps, pts_unit='sec'
        )[0][:-1] # [T, H, W, C] # [:-1] remove the close interval final frame
    assert video.shape[1] >= image_size[1] and video.shape[2] >= image_size[0], "Discard video"
    video = uniform_temporal_subsample(video, num_samples=num_frames, temporal_dim=0)

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2) # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, num_frames, fps, inference=False, filters=None):
    original_fps = fps
    for r in src:
        # read Image
        if 'mp4' in r:
            video_data = r['mp4']
        elif 'avi' in r:
            video_data = r['avi']
        else:
            print('No video data found')
            continue

        # handle filters
        filter_flag = 0
        if filters is None:
            filters = []
        for filter in filters:
            key = filter['key']
            default_score = -float('inf') if filter["greater"] else float('inf')
            score = r.get(key, default_score) or default_score
            judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
            if not judge(score):
                filter_flag = 1
                break
        if filter_flag:
            continue

        # fetch duration
        duration = r.get('duration', None)
        if duration is None:
            try:
                item = json.loads(r['json'])
                duration = item['duration']
            except Exception as e:
                print(f'Read duration error: {e}')
                continue
        duration = float(duration)

        # handle multiple fps
        if not isinstance(original_fps, float) and not isinstance(original_fps, int):
            fps = random.choice(original_fps)
        actual_fps=r.get('fps', None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        if actual_fps is None or actual_fps == 0:
            print(f'Skip zero fps video')
            continue
        if duration < num_frames / fps:
            print(f'Skip zero fps video')
            continue

        try:
            frames = process_video(io.BytesIO(video_data), num_frames=num_frames, wanted_fps=fps, image_size=image_size, duration=duration, \
                                   actual_fps=actual_fps, inference=inference)
            frames = (frames - 127.5) / 127.5
            frames = frames.permute(1, 0, 2, 3)
        except Exception as e:
            print(f'Read video error in function process_video')
            continue
        item = {
            'mp4': frames,
            'num_frames': num_frames,
            'fps': fps
        }
        yield item


class MetaDistributedWebDataset(DataPipeline):
    '''WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id'; 
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    '''
    def __init__(self, path, process_fn, seed, *, meta_names=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None):
        if include_dirs is not None: # /webdatasets/A,/webdatasets/C
            other_paths = []
            include_dirs = include_dirs.split(',')
            for include_dir in include_dirs:
                if '*' in include_dir:
                    include_dir, n = include_dir.split('*')
                    n = int(n)
                else:
                    n = 1
                for cur_dir, dirs, files in os.walk(include_dir):
                    for f in files:
                        if f.endswith('tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                            other_paths.extend([os.path.join(cur_dir,f)]*n)
            from braceexpand import braceexpand
            if len(path) > 0:
                path = list(braceexpand(path)) + other_paths
            else:
                path = other_paths

        tarfile_samples = partial(tarfile_samples_with_meta, meta_names=meta_names, handler=reraise_exception)
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # if model parallel, shuffle_buffer should be 1 to disable shuffling
        try:
            from sat.mpu import get_model_parallel_world_size
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass

        # manually set a fixed shuffle seed
        global_rank = dist.get_rank()
        rng = random.Random(seed+global_rank)
        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer, rng=rng),
            process_fn
        )


class VideoWebDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        nshards=sys.maxsize,
        seed=-1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        inference=False,
        filters=None,
        **kwargs
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []
        if path.startswith(';'):
            path, include_dirs = path.split(';', 1)
        if inference:
            shuffle_buffer = 1

        super().__init__(
            path,
            partial(process_fn_video,
                    image_size=image_size,
                    num_frames=num_frames,
                    fps=fps,
                    inference=inference,
                    filters=filters),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )

    @classmethod
    def create_dataset_function(cls, path, args, meta_names, image_size, num_frames, fps, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None
    
        return cls(path, image_size=image_size, include_dirs=include_dirs, num_frames=num_frames, fps=fps, meta_names=meta_names, **kwargs)


def resize_for_rectangle_crop(arr, image_size, reshape_mode='random'):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(arr, size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])], interpolation=InterpolationMode.BICUBIC)
    else:
        arr = resize(arr, size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]], interpolation=InterpolationMode.BICUBIC)

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == 'random' or reshape_mode == 'none':
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == 'center':
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr


def process_fn_image(src, image_size, transform, reshape_mode='random', filters=None):
    for r in src:
        # read Image
        if ('png' not in r and 'jpg' not in r):
            continue

        filter_flag = 0
        if filters is None:
            filters = []
        for filter in filters:
            key = filter['key']
            default_score = -float('inf') if filter["greater"] else float('inf')
            score = r.get(key, default_score) or default_score
            judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
            if not judge(score):
                filter_flag = 1
                break
        if filter_flag:
            continue

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        w, h = img.size
        if w < h:
            continue
        # avoid low resolution upsample to high resolution
        if w < image_size[0] or h < image_size[1]:
            continue
        arr = transform(img)

        arr = arr.unsqueeze(0)
        arr = resize_for_rectangle_crop(arr, image_size, reshape_mode=reshape_mode)
        arr = arr * 2 - 1

        arr = arr.unsqueeze(1)
        item = {
            'mp4': arr,
            'num_frames': 1,
            'fps': 0
        }
        yield item


class ImageWebDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        nshards=sys.maxsize,
        seed=-1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        filters=None,
        **kwargs
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []
        if path.startswith(';'):
            path, include_dirs = path.split(';', 1)

        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        chained_trainsforms = TT.Compose(chained_trainsforms)

        super().__init__(
            path,
            partial(process_fn_image,
                    image_size=image_size,
                    transform=chained_trainsforms,
                    filters=filters),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )


class IVAlterstepDataset(IterableDataset):
    def __init__(
        self,
        image_ds_config,
        video_ds_config,
        image_batch_size,
        video_batch_size,
        sampling_in_dataset=False,
        image_video_weights=[1, 1],
        seed=0,
    ):
        super().__init__()

        self.image_ds = instantiate_from_config(image_ds_config)
        self.video_ds = instantiate_from_config(video_ds_config)
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.sampling_in_dataset = sampling_in_dataset
        if sampling_in_dataset:
            # Fix image/video sampling by seed
            global_rank = dist.get_rank()
            rng = np.random.default_rng(seed=[seed+global_rank])
            self.rng = rng
            self.image_video_weights = np.array(image_video_weights) / sum(image_video_weights)

    def __iter__(self):
        image_iter = iter(self.image_ds)
        video_iter = iter(self.video_ds)

        def get_image_batch(image_iter):
            image_batch = []
            for _ in range(self.image_batch_size):
                image_batch.append(next(image_iter))
            image_batch = default_collate(image_batch)
            return image_batch

        def get_video_batch(video_iter):
            video_batch = []
            for _ in range(self.video_batch_size):
                video_batch.append(next(video_iter))
            video_batch = default_collate(video_batch)
            return video_batch

        while True:
            if self.sampling_in_dataset:
                index = self.rng.choice(2, p=self.image_video_weights)
                # image
                if index == 0:
                    image_batch = get_image_batch(image_iter)
                    yield image_batch
                else:
                    video_batch = get_video_batch(video_iter)
                    yield video_batch
            else:
                image_batch = get_image_batch(image_iter)
                video_batch = get_video_batch(video_iter)
                yield {
                    "video_batch": video_batch, 
                    "image_batch": image_batch
                }


class SingleDataset(IterableDataset):
    def __init__(
        self,
        ds_config,
        batch_size,
        mode='image',
    ):
        super().__init__()
        self.ds = instantiate_from_config(ds_config)
        self.batch_size = batch_size
        self.mode = mode
        assert mode in ['image', 'video']

    def __iter__(self):
        ds_iter = iter(self.ds)

        while True:
            batch = []
            for _ in range(self.batch_size):
                batch.append(next(ds_iter))
            batch = default_collate(batch)
            if self.mode == 'image':
                batch['mp4'] = batch['mp4'].squeeze(2)

            yield batch


def collate_straight_through(batch):
    return batch[0]


class VideoDataModuleFromConfig(LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None, prefetch_factor=4):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader)
        self.prefetch_factor = prefetch_factor
    
    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          collate_fn=collate_straight_through)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          prefetch_factor=self.prefetch_factor,
                          collate_fn=collate_straight_through)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle,
                          collate_fn=collate_straight_through)