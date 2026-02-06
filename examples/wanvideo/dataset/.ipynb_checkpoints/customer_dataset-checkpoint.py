import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
import pickle
import cv2
from utils.dwpose_draw import draw_dwpose_lmk, draw_dwpose_lmk_wild
import json
from loguru import logger
import copy
from PIL import Image
import traceback
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def apply_color_augmentation(
    tensor,
    brightness=0.0, brightness_prob=0.0,
    contrast=0.0, contrast_prob=0.0,
    saturation=0.0, saturation_prob=0.0,
    hue=0.0, hue_prob=0.0
):
    """
    对输入张量应用概率化的颜色增强（支持批量处理）
    
    参数:
        tensor: 输入张量 [B, C, H, W] (值范围需为 [0, 1])
        brightness: 亮度调整幅度 (范围 [0, 1], 例如 0.2 表示亮度变化 ±20%)
        brightness_prob: 应用亮度增强的概率 (0~1)
        contrast: 对比度调整幅度 (类似亮度)
        contrast_prob: 应用对比度增强的概率
        saturation: 饱和度调整幅度 (类似亮度)
        saturation_prob: 应用饱和度增强的概率
        hue: 色调调整幅度 (范围 [-0.5, 0.5])
        hue_prob: 应用色调增强的概率
    """
    augmented = tensor.clone()
    
    for b in range(augmented.shape[0]):
        img = augmented[b]  # [C, H, W]
        
        # 亮度增强（概率化）
        if random.random() < brightness_prob:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            img = TF.adjust_brightness(img, brightness_factor)
        
        # 对比度增强（概率化）
        if random.random() < contrast_prob:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # 饱和度增强（概率化）
        if random.random() < saturation_prob:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            img = TF.adjust_saturation(img, saturation_factor)
        
        # 色调增强（概率化）
        if random.random() < hue_prob:
            hue_factor = random.uniform(-hue, hue)
            img = TF.adjust_hue(img, hue_factor)
        
        augmented[b] = img
    
    return augmented

def resize_with_padding(input_tensor, target_h, target_w):
    """
    保持宽高比调整输入张量尺寸，不足处填充黑边（padding）
    输入格式: [B, C, H, W]
    输出格式: [B, C, target_h, target_w]
    """
    _, _, h, w = input_tensor.shape

    # 计算宽高比
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    # 判断以高度还是宽度为基准进行缩放
    if aspect_ratio > target_aspect_ratio:
        # 以宽度为基准缩放
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # 以高度为基准缩放
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # 插值调整尺寸
    resized = F.interpolate(
        input_tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )

    # 计算需要填充的像素数
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    # 对称填充（上下、左右均分填充量）
    padding = (
        pad_w // 2,          # 左
        pad_w - pad_w // 2,  # 右
        pad_h // 2,          # 上
        pad_h - pad_h // 2   # 下
    )

    # 应用填充（填充值为0）
    padded = F.pad(resized, padding, mode="constant", value=0)

    return padded

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

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

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

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

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


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item


class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)


class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths = []
        self.captions = []

        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)
                    self.video_paths.append(video_path)

                    caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                    if os.path.exists(caption_path):
                        caption = open(caption_path, "r").read().splitlines()[0]
                    else:
                        caption = ""
                    self.captions.append(caption)

    def __getitem__(self, index):
        
        decord.bridge.set_bridge("torch")

        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(start, end, (end - start) // num_frames).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                )
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(
                    end - start
                )  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                )

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        tensor_frms = (tensor_frms - 127.5) / 127.5

        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)


def get_frame_range(vr, frames, sample_rate=1,start_end=None):
    max_range = len(vr)
    min_range = 0
    if start_end is not None:
        max_range = start_end[1]
        min_range = start_end[0]
    if max_range-sample_rate*frames-1<0:
        return []

    frame_start = max_range-sample_rate*frames-1
    if frame_start<min_range:
        return []
    frame_number_start = random.randint(min_range,frame_start)
    frame_range = range(frame_number_start, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:frames]
    return frame_range_indices

def get_frame_range_2(max_range, frames, sample_rate=1):
    frame_start = max(0, max_range-sample_rate*frames-1)
    frame_number_start = random.randint(0, frame_start)
    frame_range = range(frame_number_start, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:frames]
    return frame_range_indices


def draw_dwpose_v4(render, dwpose, frame_indice):
    # input is tensor
    line_list = [[1, 2], [1, 5], [2, 3], [5, 6], [3, 4],  [6, 7]]
    colors = [[255, 0, 0],  [170, 255, 0], [0, 255, 170], [0, 255, 255], [0, 170, 255], [255, 0, 85]]
    new_render = []
    for index in range(len(frame_indice)):
        candidate = np.array(dwpose[frame_indice[index]]["candidate"])[0, :18]
        render_frame = render[index]
        H, W, C = render_frame.shape
        # candidate = candidate * np.array([W, W]).reshape(1, 2)
        thick = 20 * min(H, W) // 1080
        # thick = 20

        zero_image = np.zeros_like(render_frame).astype(np.float32)
        for c, line in enumerate(line_list):
            Y = candidate[line, 0] * float(W)
            X = candidate[line, 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), thick), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(zero_image, polygon, colors[c])
        # zero_image = torch.from_numpy(zero_image)
        # render_frame_mask = (render_frame > 10).int()
        render_frame_mask = render_frame > 10
        render_frame = render_frame + (1 - render_frame_mask) * zero_image
        new_render.append(render_frame)
    # new_render = torch.stack(new_render, 0)
    return new_render

def random_crop_videos_multi_res(video_frames, render_frames=None, target_size=None):
    # import ipdb;ipdb.set_trace()
    F, C, H, W = video_frames.shape
    top = 0
    left = 0
    target_ratio = target_size[0] / target_size[1]
    origin_ratio = H / W
    if target_ratio > origin_ratio:
        if H <= target_size[0]:
            crop_size_h = H
            top = 0
            crop_size_w = int(H / target_ratio)
            left = random.randint((W - crop_size_w - 1) // 7 * 3, (W - crop_size_w - 1) // 7 * 4)

        elif H > target_size[0]:
            if random.random() < 0.2:
                crop_size_h = H - 1
            else:
                crop_h_low_bound = target_size[0]
                while crop_h_low_bound * 1.2 < H - 1:
                    crop_h_low_bound = crop_h_low_bound * 1.2
                crop_h_low_bound = int(crop_h_low_bound)
                crop_size_h = random.randint(crop_h_low_bound, H - 1)
            top = random.randint((H - crop_size_h) // 7, (H - crop_size_h) // 7 * 3)

            crop_size_w = int(crop_size_h / target_ratio)
            left = random.randint((W - crop_size_w - 1) // 7 * 3, (W - crop_size_w - 1) // 7 * 4)
    else:
        if W <= target_size[1]:
            crop_size_w = W
            left = 0
            crop_size_h = int(W * target_ratio)
            try:
                top = random.randint((H - crop_size_h - 1) // 7, (H - crop_size_h - 1) // 7 * 4)
            except:
                top = 0
                # import ipdb;ipdb.set_trace()

        else:
            if random.random() < 0.2:
                crop_size_w = W - 1
            else:
                crop_w_low_bound = target_size[1]
                while crop_w_low_bound * 1.2 < W - 1:
                    crop_w_low_bound = crop_w_low_bound * 1.2
                crop_w_low_bound = int(crop_w_low_bound)
                crop_size_w = random.randint(crop_w_low_bound, W - 1)
            left = random.randint((W - crop_size_w - 1) // 7 * 3, (W - crop_size_w - 1) // 7 * 4)

            crop_size_h = int(crop_size_w * target_ratio)
            top = random.randint((H - crop_size_h - 1) // 7, (H - crop_size_h - 1) // 7 * 3)
    video_frames = video_frames[:, :, top:top+crop_size_h, left:left+crop_size_w]
    if render_frames is not None:
        render_frames = render_frames[:, :, top:top+crop_size_h, left:left+crop_size_w]
    if video_frames.shape[0] != target_size[0] or video_frames.shape[1] != target_size[1]:
        video_frames = torch.nn.functional.interpolate(video_frames, size=target_size, mode="bicubic", align_corners=True, antialias=True)
        if render_frames is not None:
            render_frames = torch.nn.functional.interpolate(render_frames, size=target_size, mode="bicubic", align_corners=True, antialias=True)
    video_frames = (video_frames.float() / 255.0 - 0.5) / 0.5
    if render_frames is not None:
        render_frames = (render_frames.float() / 255.0 - 0.5) / 0.5
    return video_frames, render_frames
    
def get_all_resolution(image_size):
    all_resolution = []
    divided_by = 32
    min_edge = int(image_size / 1.4)
    max_edge = int(image_size * 1.4)
    token_number = image_size * image_size / divided_by / divided_by
    for i in range(min_edge // divided_by, max_edge // divided_by + 1):
        all_resolution.append([i * divided_by, int(token_number // i * divided_by)])
    return all_resolution
        

class HumanDataset(Dataset):
    def __init__(self, data_dir="", video_size=768, fps=25, max_num_frames=7, skip_frms_num=3,ref_id_type="random"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(HumanDataset, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        if data_dir is None or data_dir == "":
            video_list_path = "data/training_0801.list"
        else:
            video_list_path = data_dir

        videos_list = []
        with open(video_list_path, 'r') as f:
            for line in f.readlines():
                videos_list.append(line.strip())
        self.videos_list = videos_list
        random.shuffle(self.videos_list)

        image_size_list = [x * 8 for x in range(64, 97, 2)]
        # image_size_list = [x * 8 for x in range(96, 97, 2)]
        # frame_number_list = [int(768. * 768. / x / x * 7) for x in image_size_list]
        # frame_number_list = [int(768. * 768. / x / x * 17) for x in image_size_list]
        frame_number_list = [int(768. * 768. / x / x * 33) for x in image_size_list]
        for i in range(len(frame_number_list)):
            # frame_number_list[i] = frame_number_list[i] // 4 * 4 + 1
            frame_number_list[i] = frame_number_list[i] // 4 * 4 + 1
        probs = [x+1 for x in range(len(frame_number_list))]
        self.probs = [float(x) / sum(probs) for x in probs]
        self.frame_number_list = frame_number_list
        self.image_size_list = image_size_list

        # self.image_size_list = [512]
        # self.frame_number_list = [17]
        # print("frame_number_list:", self.frame_number_list)
        # print("image_size_list:", self.image_size_list)


    def __getitem__(self, index):
        # decord.bridge.set_bridge("torch")
        while True:
            try:
                video_index = random.randint(0, len(self.videos_list) - 1)
                video_path = self.videos_list[video_index]
                render_path = video_path.replace('/videos/', '/renders/')
                info_path = video_path.replace('/videos/', '/hamer_hifi/').replace('.mp4', '.pickle')
                video = decord.VideoReader(video_path)


    
                size_index = np.random.choice(range(len(self.image_size_list)), p=self.probs)
                resolution_list = get_all_resolution(self.image_size_list[size_index])
                frame_num = self.frame_number_list[size_index]
                # TODO 
                # res_prob = [x+1 for x in range(len(resolution_list))]
                # res_prob = [float(x) / sum(res_prob) for x in res_prob]

                # cur_resolution_index = np.random.choice(len(resolution_list), p=res_prob)
                cur_resolution_index = np.random.choice(len(resolution_list))
                cur_resolution = resolution_list[cur_resolution_index]

                # frame_num = 1
                # sample_rate = random.randint(1, 2)
                sample_rate = 1

                frame_indice = get_frame_range(video, frames=frame_num, sample_rate=sample_rate)
                
                if len(frame_indice) != frame_num:
                    print(f'{video_path} not enough frames')
                    continue

                render = decord.VideoReader(render_path)
                rand_index = random.randint(0, len(video) - 1)
                frame_indice.append(rand_index)

                if len(video) != len(render):
                    print(f'{video_path} video and render not same length')
                    continue
                video_frames = video.get_batch(frame_indice).asnumpy() # (N, H, W, C)
                render_frames = render.get_batch(frame_indice).asnumpy() # (N, H, W, C)
                
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                render_frames = draw_dwpose_v4(render_frames, infos["dwpose"], frame_indice) # (N, H, W, C)

                # video_frames = torch.permute(video_frames, (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                # render_frames = torch.permute(render_frames, (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                render_frames = torch.permute(torch.tensor(np.array(render_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

                video_frames, render_frames = random_crop_videos_multi_res(video_frames, render_frames, cur_resolution)

                item = {
                    "video_frames": video_frames,
                    "render_frames": render_frames,
                    "num_frames": frame_num,
                    "fps": 25 // sample_rate,
                }
                return item
            except:
                del self.videos_list[video_index]
                print(f'data len: {len(self.videos_list)} load {video_path} failed!')
                continue

    def __len__(self):
        return len(self.videos_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)


class HumanMotionDataset(Dataset):
    def __init__(self, data_dir="", video_size=768, fps=25, max_num_frames=7, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(HumanMotionDataset, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        if data_dir is None or data_dir == "":
            video_list_path = "data/training_0801.list"
        else:
            video_list_path = data_dir

        videos_list = []
        with open(video_list_path, 'r') as f:
            for line in f.readlines():
                videos_list.append(line.strip())
        self.videos_list = videos_list
        random.shuffle(self.videos_list)

        if len(self.videos_list) < 1000:
            self.ft_single=True
            image_size_list = [x * 8 for x in range(96, 97, 2)]
        else:
            self.ft_single=False
            image_size_list = [x * 8 for x in range(64, 97, 2)]
        # image_size_list = [x * 8 for x in range(32, 48, 2)]
        # image_size_list = [x * 8 for x in range(48, 65, 2)]

        frame_number_list = [int(768. * 768. / x / x * 25) for x in image_size_list]
        # frame_number_list = [int(384. * 384. / x / x * 25) for x in image_size_list]
        for i in range(len(frame_number_list)):
            frame_number_list[i] = frame_number_list[i] // 4 * 4 + 1 + 5
        probs = [x+1 for x in range(len(frame_number_list))]
        self.probs = [float(x) / sum(probs) for x in probs]
        self.frame_number_list = frame_number_list
        self.image_size_list = image_size_list
        print("frame_number_list:", frame_number_list)


    def __getitem__(self, index):
        # decord.bridge.set_bridge("torch")
        while True:
            try:
                video_index = random.randint(0, len(self.videos_list) - 1)
                video_path = self.videos_list[video_index]

                render_path = video_path.replace('/videos/', '/renders/')
                info_path = video_path.replace('/videos/', '/hamer_hifi/').replace('.mp4', '.pickle')
                video = decord.VideoReader(video_path)

                size_index = np.random.choice(range(len(self.image_size_list)), p=self.probs)
                resolution_list = get_all_resolution(self.image_size_list[size_index])
                frame_num = self.frame_number_list[size_index]
                # TODO 
                frame_num = frame_num # motion length

                # res_prob = [x+1 for x in range(len(resolution_list))]
                # res_prob = [float(x) / sum(res_prob) for x in res_prob]

                # cur_resolution_index = np.random.choice(len(resolution_list), p=res_prob)
                cur_resolution_index = np.random.choice(len(resolution_list))
                cur_resolution = resolution_list[cur_resolution_index]

                # frame_num = 1
                # sample_rate = random.randint(1, 2)
                sample_rate = 1

                frame_indice = get_frame_range(video, frames=frame_num, sample_rate=sample_rate)
                
                if len(frame_indice) != frame_num:
                    print(f'{video_path} not enough frames')
                    continue

                render = decord.VideoReader(render_path)
                rand_index = random.randint(0, len(video) - 1)
                frame_indice.append(rand_index)

                if len(video) != len(render):
                    print(f'{video_path} video and render not same length')
                    continue

                video_frames = video.get_batch(frame_indice).asnumpy() # (N, H, W, C)
                render_frames = render.get_batch(frame_indice).asnumpy() # (N, H, W, C)

                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                render_frames = draw_dwpose_v4(render_frames, infos["dwpose"], frame_indice) # (N, H, W, C)
                if self.ft_single:
                    image_size = 768
                    new_shape = get_resize_shape_v2(video_frames[0].shape, image_size)
                    video_frames = transforms_videos(video_frames, new_shape) # numpy(N, H, W, C) 0~255 -> torch(N, C, H, W) -1~1
                    render_frames = transforms_videos(render_frames, new_shape)
                    video_frames = fix_crop_video(video_frames, image_size)
                    render_frames = fix_crop_video(render_frames, image_size)
                else:
                    video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                    render_frames = torch.permute(torch.tensor(np.array(render_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                    video_frames, render_frames = random_crop_videos_multi_res(video_frames, render_frames, cur_resolution)
                item = {
                    "video_frames": video_frames,
                    "render_frames": render_frames,
                    "num_frames": frame_num,
                    "fps": 25 // sample_rate,
                }
                return item
            except:
                del self.videos_list[video_index]
                print(f'data len: {len(self.videos_list)} load {video_path} failed!')
                continue

    def __len__(self):
        return len(self.videos_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

def mask_strategy(video_frames, wo_obj_video_frames, pixel_values_mesh, pixel_values_box, hand_masks):
    rand_flag = random.random()
    inpaint_num_list = [1,2,3]
    box_num_list = [1,2,3]
    mesh_num_list = [1,2,3]
    if rand_flag <  0.1:
        inpaint_num = random.choice(inpaint_num_list)
        inpaint_idx = random.sample(range(9, 38), inpaint_num) 
        wo_obj_video_frames[inpaint_idx] = video_frames[inpaint_idx]

    elif 0.1 <= rand_flag < 0.4:
        # import ipdb;ipdb.set_trace()
        box_num = random.choice(box_num_list)
        mesh_num = random.choice(mesh_num_list)
        all_idx = random.sample(range(9, 38), (box_num+mesh_num))
        box_idx = all_idx[:box_num]
        mesh_idx = all_idx[box_num:]

        pixel_values_box[box_idx] = 0.
        pixel_values_mesh[mesh_idx] = 0.
        hand_masks[mesh_idx] = 0.
        # wo_obj_video_frames[box_idx] = video_frames[box_idx]
        # wo_obj_video_frames[mesh_idx] = video_frames[mesh_idx]
    else:
        pass
    
    return wo_obj_video_frames, pixel_values_mesh, pixel_values_box,hand_masks
        
        




class HumanHoiDataset(Dataset):
    def __init__(self, data_dir="", video_size=768, fps=25, max_num_frames=7, skip_frms_num=3,ref_id_type="random", draw_hand_color="ori", info_class="dwpose_test123", data_aug = True):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(HumanHoiDataset, self).__init__()
        
        self.video_size = video_size #[768, 768]
        self.fps = fps
        self.max_num_frames = max_num_frames # 33
        self.skip_frms_num = skip_frms_num   # 3.0
        self.aug = data_aug

        if data_dir is None or data_dir == "":
            video_list_path = "data/training_0801.list"
        else:
            video_list_path = data_dir

        # import ipdb;ipdb.set_trace()

        ### load video list
        with open(video_list_path, 'r') as csvfile:
            import csv
            self.videos_list = list(csv.DictReader(csvfile))
            random.shuffle(self.videos_list)

        image_size_list = [x * 8 for x in range(96, 97, 2)]
        image_size_list = [512]
        frame_number_list = [int(768. * 768. / x / x * 17) for x in image_size_list]
        #frame_number_list = [int(512. * 512. / x / x * 17) for x in image_size_list]
        for i in range(len(frame_number_list)):
            frame_number_list[i] = 33 + 5
        probs = [x+1 for x in range(len(frame_number_list))]
        self.probs = [float(x) / sum(probs) for x in probs]
        self.frame_number_list = frame_number_list
        self.image_size_list = image_size_list
        self.ref_id_type = ref_id_type
        self.draw_hand_color = draw_hand_color
        self.info_class = info_class

        self.data_root = "/root/paddlejob/workspace/shenzhelun/wan/"
        print('ref_id_type:',ref_id_type)
        print("frame_number_list:", frame_number_list)
        print(f"draw_hand_color:{draw_hand_color}")


    def __getitem__(self, index):
        while True:
            try:
                video_index = random.randint(0, len(self.videos_list) - 1)
                if "\t" in self.videos_list[video_index]:
                    video_path, start_end = self.videos_list[video_index].split("\t")
                    start,end = start_end.split(":")
                    start_end = [int(start),int(end)]

                else:
                    data_dict = self.videos_list[video_index]
                    start_end = None
                # print("train:", video_path)
                
                gt_path, pose_path, obj_mask_path, obj_path = \
                     data_dict['video_path'], data_dict['pose_path'], data_dict['obj_mask_path'], data_dict['obj_video_path']

                gt_path = os.path.join(self.data_root, gt_path)
                pose_path = os.path.join(self.data_root, pose_path)
                obj_mask_path = os.path.join(self.data_root, obj_mask_path)
                obj_path = os.path.join(self.data_root, obj_path)
                txt_path = gt_path.replace(".mp4", ".txt")
                with open(txt_path, 'r') as f:
                        prompt = f.read().strip()

                
                video = decord.VideoReader(gt_path)
                fps = int(video.get_avg_fps())
                if start_end is not None:
                    start_end = [ int(x * fps) for x in start_end   ]
                
                # import ipdb;ipdb.set_trace()
                size_index = np.random.choice(range(len(self.image_size_list)), p=self.probs)
                resolution_list = get_all_resolution(self.image_size_list[size_index])
                # frame_num = self.frame_number_list[size_index]
                frame_num = self.max_num_frames
                # TODO 
                # frame_num = frame_num # motion length all length
                # import ipdb;ipdb.set_trace()
                # cur_resolution_index = np.random.choice(len(resolution_list))
                # cur_resolution = resolution_list[cur_resolution_index]
                cur_resolution = self.video_size ## hard code, revise it later
                sample_rate = 1

                
                ## motion frames + gt frames
                frame_indice = get_frame_range(video, frames=frame_num, sample_rate=sample_rate, start_end=start_end)
                if len(frame_indice) != frame_num:
                    print(f'{gt_path} not enough frames')
                    continue
                video_frames = video.get_batch(frame_indice).asnumpy() # (N, H, W, C)
                render_frames = np.zeros_like(video_frames)
                H, W = video_frames.shape[1], video_frames.shape[2]
                width, height = W, H

                video_frames  = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                wo_obj_video_frames = video_frames.clone()


                ## ref frame generation
                obj_video = decord.VideoReader(obj_path)
                ref_idx = random.randint(0, len(obj_video) - 1)
                ref_img = obj_video[ref_idx].asnumpy() # (N, H, W, C)
                pixel_values_ref_img = torch.permute(torch.tensor(np.array(ref_img)), (2, 0, 1)).unsqueeze(0)
                pixel_values_ref_img = resize_with_padding(pixel_values_ref_img, target_h=video_frames.shape[2], target_w=video_frames.shape[3])
                print(pixel_values_ref_img.shape)

                if random.random() < 0.5 and self.aug:
                    pixel_values_ref_img = apply_color_augmentation(
                        pixel_values_ref_img,
                        brightness=0.2, brightness_prob=0.5,   # 50% 概率调整亮度
                        contrast=0.2, contrast_prob=0.5,       # 50% 概率调整对比度
                        saturation=0.2, saturation_prob=0.5,   # 50% 概率调整饱和度
                        hue=0.05, hue_prob=0.5                 # 50% 概率调整色调
                    )

                if random.random() < 0.5 and self.aug:
                    wo_obj_video_frames = apply_color_augmentation(
                        wo_obj_video_frames,
                        brightness=0.2, brightness_prob=0.5,   # 50% 概率调整亮度
                        contrast=0.2, contrast_prob=0.5,       # 50% 概率调整对比度
                        saturation=0.2, saturation_prob=0.5,   # 50% 概率调整饱和度
                        hue=0.05, hue_prob=0.5                 # 50% 概率调整色调
                    )
                ## Layout and mesh Pose and img inpaint
                obj_masks = decord.VideoReader(obj_mask_path)
                with open(pose_path, "r") as f:
                    gt_2dpose = json.load(f)    # json keys: ['crop_video_path', 'box', 'width', 'height', 'dwpose']
                
                # kps = [pose["candidate"][0] for pose in gt_2dpose["dwpose"]]    # ['candidate', 'subset' (score), 'box', 'size']
                # scores = [pose["subset"][0] for pose in gt_2dpose["dwpose"]]
                # scores = np.array(scores) #N,134
                # kps = np.array(kps)       #N,134,2

                # import ipdb;ipdb.set_trace()
                rtmw = True
                render_frames = draw_dwpose_lmk_wild(render_frames, gt_2dpose['dwpose'], frame_indice, rtmw=rtmw, draw_hand_color=self.draw_hand_color, only_hand=True) # (N, H, W, C)
                render_frames = torch.permute(torch.tensor(np.array(render_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

                layout_path = gt_path.replace("clip_videos", "layout")
                hand_mask_path = gt_path.replace("clip_videos", "hand_mask")

                # import ipdb;ipdb.set_trace()
                vr_layout = decord.VideoReader(layout_path)
                vr_hand_mask = decord.VideoReader(hand_mask_path)

                pixel_values_box  = vr_layout.get_batch(frame_indice).asnumpy()
                hand_masks = vr_hand_mask.get_batch(frame_indice).asnumpy()

                pixel_values_box = torch.from_numpy(np.array(pixel_values_box)).permute(0, 3, 1, 2)
                hand_masks = torch.from_numpy(np.array(hand_masks)).permute(0, 3, 1, 2)/255.0
                
                pixel_values_mesh = render_frames*hand_masks
                

                
                for i,idx in enumerate(frame_indice):
                    img_mask = obj_masks[idx].asnumpy()
                    img_mask = np.array(img_mask) / 255
                    mask_idx = np.where(img_mask == 1)

                    # if mask_idx==None:
                    #     continue
                    try:
                        h1,h2,w1,w2 = min(mask_idx[0]),max(mask_idx[0]),min(mask_idx[1]),max(mask_idx[1])
                        h = h2-h1
                        w = w2-w1
                        center_h = (h1+h2)//2
                        center_w = (w1+w2)//2
                        mask_size = int((max(h,w)*1.25)//2 * 2)
                        # import ipdb;ipdb.set_trace()

                        wo_obj_video_frames[i,:,center_h-(mask_size//2):center_h+(mask_size//2),center_w-(mask_size//2):center_w+(mask_size//2)] = 0
                    except:
                        pass
                        # print("no obj mask",obj_mask_path)


                hand_masks = hand_masks*255.

                cond_arguement = True
                if cond_arguement:
                    wo_obj_video_frames, pixel_values_mesh, pixel_values_box, hand_masks = \
                            mask_strategy(video_frames, wo_obj_video_frames, pixel_values_mesh, pixel_values_box, hand_masks)

                print("shape of video frams", video_frames.shape)
                video_frames  = torch.cat([video_frames, wo_obj_video_frames, pixel_values_ref_img], dim=0) ###　38,38,3
                render_frames = torch.cat([pixel_values_mesh, pixel_values_box, hand_masks], dim=0) #0 or 1
                #render_frames = torch.cat([pixel_values_mesh, pixel_values_box], dim=0) #0 or 1
                ## pixel_values_mesh the pose of the hand
                ## pixel_values_box three boundingbox
                ## hand mask, the mask of the hand  
                ## video_frames the gt video
                ## wo_obj_video_frames  the video masked the object and hand
                ## pixel_values_ref_img the ref frame of the object


                # render_frames = torch.permute(torch.tensor(np.array(render_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                video_frames, render_frames = random_crop_videos_multi_res(video_frames, render_frames, cur_resolution)

                gt_frames = video_frames[:self.max_num_frames, :, :, :]
                wo_obj_video_frames = video_frames[self.max_num_frames:self.max_num_frames * 2, :, :, :]
                pixel_values_ref_img = video_frames[self.max_num_frames * 2:, :, :, :]
                hand_pose = render_frames[ :self.max_num_frames, :, :, :]
                hand_obj_box = render_frames[self.max_num_frames:self.max_num_frames * 2, :, :, :]
                # print("render_frames:",render_frames.shape)
                # import ipdb;ipdb.set_trace() #-1，1
                # import pdb;pdb.set_trace()
                item = {
                    "gt_frames": gt_frames.contiguous(),
                    "wo_obj_video_frames": wo_obj_video_frames.contiguous(),
                    "pixel_values_ref_img": pixel_values_ref_img.contiguous(),
                    "hand_pose": hand_pose.contiguous(),
                    "hand_obj_box": hand_obj_box.contiguous(),
                    "num_frames": frame_num,
                    "prompt": prompt,
                    "fps": 25 // sample_rate,
                }
                # item = {
                #     "video_frames": video_frames.contiguous(),
                #     "render_frames": render_frames.contiguous(),
                #     "num_frames": frame_num,
                #     "fps": 25 // sample_rate,
                # }
                return item
            except:
                # del self.videos_list[video_index]
                print(f'load {gt_path} failed!')
                print(sys.exc_info())
                traceback.print_exc()  # 打印完整错误信息，包括行号
                continue

    def __len__(self):
        return len(self.videos_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)




def read_dwpose_test_data(video_path, frame_indice, image_size=768,info_class="rtmw_pose",info_path=None,video_frame_indice=None,):
    """
    video_path :原视频mp4路径，真人视频，从其中读物rgb frames，用于motion_frames等
    frame_indice:要生成的indice，用于从dwpose读取相应帧的信息画pose,根据frame_indice画pose。
    image_size: 分辨率
    info_class: 读取哪一类dwpose
    info_path: 指定dwpose的路径，不按video_path去都去,用于测试新的dwpose
    video_frame_indice:指定rgb 视频帧读取。
    
    """
    # decord.bridge.set_bridge("torch")
    if info_path is None:
        if info_class in ["rtmw_pose","sapiens_2b"]:
            info_path = video_path.replace('/videos/', f'/{info_class}/').replace('.mp4', '.json')
            infos = json.load(open(info_path, 'r'))
        elif info_class in ["hamer_hifi"]:
            info_path = video_path.replace('/videos/', f'/{info_class}/').replace('.mp4', '.pickle')
            infos = pickle.load(open(info_path, 'rb'))
    else:
        if info_path.endswith(".pickle"):
            infos = pickle.load(open(info_path, 'rb'))
        elif info_path.endswith(".json"):
            infos = json.load(open(info_path, 'r'))

    video = decord.VideoReader(video_path)
    
    if video_frame_indice is None:
        video_frames = video.get_batch(list(range(0,len(video)))  ).asnumpy() # (N, H, W, C)
    else:
        logger.info(f"video_frame_indice:{video_frame_indice}")
        video_frame_indice = [min(x,len(video)-1) for x in video_frame_indice ]
        video_frames = video.get_batch(video_frame_indice).asnumpy()
        logger.info(f"video_frame_indice:{video_frame_indice}")
    new_shape = get_resize_shape_v2(video_frames[0].shape, image_size)

    # render_frames = np.zeros_like(video_frames)
    F,H,W,C = video_frames.shape
    render_frames = np.zeros((len(frame_indice),H,W,C))
    # if info_class in ["rtmw_pose","sapiens_2b"]:
    if True:
        use_rtmw = 'rtmw' in info_class
        logger.info(f"{info_path} use_rtmw: {use_rtmw}")
        render_frames = draw_dwpose_lmk_wild(render_frames,infos['dwpose'],frame_indice,rtmw=use_rtmw,draw_hand_color='ori')
    else:
        render_frames = draw_dwpose_lmk(render_frames, infos['dwpose'], frame_indice) # (N, H, W, C)

    video_frames = transforms_videos(video_frames, new_shape) # numpy(N, H, W, C) 0~255 -> torch(N, C, H, W) -1~1
    render_frames = transforms_videos(render_frames, new_shape)

    video_frames = fix_crop_video(video_frames, image_size)
    render_frames = fix_crop_video(render_frames, image_size)

    return video_frames, render_frames

def load_dwpose_info(info_path):
    if info_path.endswith(".json"):
        return json.load(open(info_path, 'r'))
    elif info_path.endswith(".pickle"):
        return pickle.load(open(info_path, 'rb'))


def keep_video_crop(video_frames, image_size):
    all_res = get_all_resolution(image_size)
    N, C, H, W = video_frames.shape
    # print("video_frames:", video_frames.shape)
    new_H = all_res[0][0]
    new_W = all_res[0][1]
    for i in range(len(all_res)):
        if H / W > all_res[i][0] / all_res[i][1]:
            new_H = all_res[i][0]
            new_W = all_res[i][1]
    # print("new_H:", new_H)
    # print("new_W:", new_W)
    if H / W != new_H / new_W:
        crop_H = int(new_H / new_W * W)
        # up = (H - crop_H) // 2
        up = (H - crop_H)
        # print("crop_H:", crop_H)
        # print("up:", up)
        video_frames = video_frames[:, :, up:up+crop_H]
        # print("video_frames0:", video_frames.shape)
    else:
        up = None
        crop_H = None

    
    video_frames = torch.nn.functional.interpolate(video_frames, size=(new_H, new_W), mode="bicubic", align_corners=True, antialias=True)
    print("video_frames1:", video_frames.shape)
    crop_info = {
        "up":up,
        "crop_H": crop_H,
        "new_H": new_H,
        "new_W": new_W,
    }
    return video_frames,crop_info


def fix_crop_video(video, max_size,center_crop=False):
    B, C, H, W = video.shape
    up = None
    left = None
    if H == max_size and W == max_size:
        return video
    elif H == max_size:
        left = (W - max_size - 1) // 2
        video = video[:, :, :, left:left + max_size]
    else:
        # up = (H - max_size - 1) // 2
        # up = 0
        up = (H - max_size - 1) // 2 if center_crop else 0
        video = video[:, :, up:up + max_size, :]
    crop_info = {
        "up":up,
        "left":left,
    }
    return video,crop_info


def transforms_videos(video_frames, target_size, interpolation_mode="bicubic"):
    video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
    video_frames = torch.nn.functional.interpolate(video_frames, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)
    video_frames = (video_frames.float() / 255.0 - 0.5) / 0.5
    return video_frames



def transforms_videos_hoi(video_frames, target_size, interpolation_mode="bicubic"):
    video_frames = torch.nn.functional.interpolate(video_frames, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)
    video_frames = (video_frames.float() / 255.0 - 0.5) / 0.5
    return video_frames


def transforms_videos_raw(video_frames, target_size=None, interpolation_mode="bicubic"):
    video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
    video_frames = (video_frames.float() / 255.0 - 0.5) / 0.5
    return video_frames

crop_func_dict={
    "keep_video_crop": keep_video_crop,
    "fix_crop_video": fix_crop_video,
}

transform_func_dict={
    "transforms_videos": transforms_videos,
    "transforms_videos_raw": transforms_videos_raw,
}

def read_video_tensors(video_path,frame_indice=None):
    video = decord.VideoReader(video_path)
    if frame_indice is None:
        frame_indice = list(range(0,len(video)))
    video_frames = video.get_batch(frame_indice).asnumpy()
    # video_frames = transforms_videos_raw(video_frames)
    video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2))
    return video_frames

def get_keep_crop_dict(video_frames,image_size):
    '''
    crop  by nearest ratio 
    '''
    all_res = get_all_resolution(image_size)
    N,C,H,W = video_frames.shape
    new_H = all_res[0][0]
    new_W = all_res[0][1]
    for i in range(len(all_res)):
        if H / W > all_res[i][0] / all_res[i][1]:
            new_H = all_res[i][0]
            new_W = all_res[i][1]
    if H / W != new_H / new_W:
        crop_H = int(new_H / new_W * W)
        up = (H - crop_H)
        video_frames = video_frames[:, :, up:up+crop_H]
    else:
        up = None
        crop_H = None
    return {
        "up":up,
        "crop_H": crop_H,
        "left":0,
        "crop_W": W,
        "new_H": new_H,
        "new_W": new_W,
    }

def get_fix_crop_dict(video,max_size,center_crop_h = False):
    '''
    center crop
    '''
    B,C,H,W = video.shape
    if H > W:
        if center_crop_h:
            up = (H-W)//2
            
        else:
            up = H - W
        left = 0
        crop_H = W
        crop_W = W
        new_H = max_size
        new_W = max_size
    else:
        up = 0
        left = (W-H) // 2
        crop_H = H
        crop_W = H
        new_H = max_size
        new_W = max_size

    return {
        "up":up,
        "crop_H": crop_H,
        "left":left,
        "crop_W": crop_W,
        "new_H": new_H,
        "new_W": new_W,
    }


def crop(video,crop_box):
    x1,x2,y1,y2 = crop_box[:]
    # print(x1,x2,y1,y2)
    # print(type(x1),type(x2),type(y1),type(y2))
    # print(video.shape)
    return video[:,y1:y2,x1:x2]


def read_video_from_ind_new(video_path,frame_indice=None,image_size=768,crop_box=None,crop_type="keep_crop",last_video_path=None):
    video = decord.VideoReader(video_path)
    if frame_indice is None:
        frame_indice = list(range(0,len(video)))
    last_frame_indice = [x for x in frame_indice if x <0]
    current_frame_indice = [x for x in frame_indice if x >=0]
    # video_frames = video.get_batch(frame_indice).asnumpy()
    video_frames = video.get_batch(current_frame_indice).asnumpy()
    if len(last_frame_indice) is not None:
        last_video = decord.VideoReader(last_video_path)
        last_video_frames = last_video.get_batch(last_frame_indice).asnumpy()
        video_frames = np.concatenate([last_video_frames,video_frames],0)

    ori_video_shape = video_frames.shape[1:]
    # new_shape = get_resize_shape_v2(video_frames[0].shape, image_size)
    if crop_box:
        print(f"crop_box:{crop_box}")
        video_frames = crop(video_frames,crop_box) #对原始视频进行 crop 预处理
    video_tensors = transforms_videos_raw(video_frames) #转tensor
    
    if crop_type == "keep_crop":
        crop_resize_info = get_keep_crop_dict(video_tensors,image_size)
    else:
        crop_resize_info = get_fix_crop_dict(video_tensors,image_size)
    up,left,crop_H,crop_W,new_H,new_W = [crop_resize_info[k] for k in ["up","left","crop_H","crop_W", "new_H","new_W"]]
    video_frames = torch.nn.functional.interpolate(video_tensors[:,:,up:up+crop_H,left:left+crop_W],size=(new_H,new_W),mode='bicubic',align_corners=True,antialias=True)
    crop_info = {
        "pre_crop_box": crop_box,
        "crop_info": crop_resize_info,
        'ori_video_shape': ori_video_shape
    }
    return video_frames,crop_info,

def draw_dwpose_from_ind_new(info_path,frame_indice,image_size=768,ori_video_shape=None, crop_box=None,crop_type="keep_crop", info_class="rtmw_pose",last_info_path=None):
    infos = load_dwpose_info(info_path)
    H,W,C = ori_video_shape[:]
    last_frame_indice = [x for x in frame_indice if x <0]
    current_frame_indice = [x for x in frame_indice if x >=0]

    render_frames = np.zeros((len(current_frame_indice),H,W,C))
    if len(last_frame_indice) > 0:
        last_infos = load_dwpose_info(last_info_path)
        last_render_frames = np.zeros((len(last_frame_indice),H,W,C))


    if info_class in ["rtmw_pose","sapiens_2b"]:
        use_rtmw = rtmw='rtmw' in info_class
        render_frames = draw_dwpose_lmk_wild(render_frames,infos['dwpose'],current_frame_indice,rtmw=use_rtmw,draw_hand_color='ori')
        if len(last_frame_indice)>0:
            last_render_frames = draw_dwpose_lmk_wild(last_render_frames,last_infos['dwpose'],last_frame_indice,rtmw=use_rtmw,draw_hand_color='ori')
            render_frames = last_render_frames + render_frames
    else:
        render_frames = draw_dwpose_lmk(render_frames, infos['dwpose'], current_frame_indice) # (N, H, W, C)
        if len(last_frame_indice)>0:
            last_render_frames = draw_dwpose_lmk(last_render_frames,last_infos['dwpose'],last_frame_indice,)
            render_frames = last_render_frames + render_frames

    render_frames = np.stack(render_frames,0)
    if crop_box:
        render_frames = crop(render_frames,crop_box)
    render_frames = transforms_videos_raw(render_frames)
    if crop_type == "keep_crop":
        crop_resize_info = get_keep_crop_dict(render_frames,image_size)
    else:
        crop_resize_info = get_fix_crop_dict(render_frames,image_size)
    up,left,crop_H,crop_W,new_H,new_W = [crop_resize_info[k] for k in ["up","left","crop_H","crop_W", "new_H","new_W"]]
    render_frames = torch.nn.functional.interpolate(render_frames[:,:,up:up+crop_H,left:left+crop_W],size=(new_H,new_W),mode='bicubic',align_corners=True,antialias=True)
    return render_frames


def read_video_from_ind(video_path,frame_indice=None,image_size=768,crop_kwargs={}):
    video = decord.VideoReader(video_path)
    if frame_indice is None:
        frame_indice = list(range(0,len(video)))


    last_frame_indice = [x for x in frame_indice if x <0]
    current_frame_indice = [x for x in frame_indice if x >=0]
    crop_info = {}
    
    video_frames = video.get_batch(current_frame_indice).asnumpy()
    if len(last_frame_indice)>0:
        video_name = video_path.split('/')[-1]
        video_index = int(video_name.split('_')[0])
        last_video_path = video_path.replace(video_name,f"{video_index-10}_{video_index}.mp4")
        if not os.path.exists(last_video_path):
            raise ValueError(f"last_video_path:{last_video_path} not exists!")
        last_video = decord.VideoReader(last_video_path)
        last_frames = last_video.get_batch(last_frame_indice).asnumpy()
        video_frames = np.concatenate([last_frames,video_frames],0)
    logger.info(f"video_path:{video_path} {video_frames.shape}")

    ori_video_shape = video_frames.shape[1:]
    new_shape = get_resize_shape_v2(video_frames[0].shape, image_size)
    crop_info["ori_shape"] = ori_video_shape
    crop_info["new_shape"] = new_shape

    transform_func_key = crop_kwargs.get("transform_function","transforms_videos")
    crop_func_key = crop_kwargs.get('crop_function','fix_crop_video')
    logger.info(f"transform_key:{transform_func_key} crop_key:{crop_func_key}")

    transforms_func = transform_func_dict[transform_func_key]
    video_frames = transforms_func(video_frames,new_shape)

    crop_func = crop_func_dict[crop_func_key]
    if 'crop_hook' in crop_kwargs.keys():
        video_frames =  crop_kwargs['crop_hook'](video_frames)
    video_frames,crop_info_2 =  crop_func(video_frames,image_size,**(crop_kwargs.get('crop_function_kwargs',{})))
    crop_info.update(crop_info_2)
    return video_frames,crop_info

def draw_dwpose_from_ind(info_path,frame_indice,image_size=768,ori_vid_shape=None,info_class="rtmw_pose",crop_kwargs={}):
    last_frame_indice = [x for x in frame_indice if x <0]
    current_frame_indice = [x for x in frame_indice if x >=0]

    if len(last_frame_indice)>0:
        json_name = info_path.split('/')[-1]
        ext = json_name.split('.')[-1]
        json_index = int(json_name.split('_')[0])
        last_json_path = info_path.replace(json_name,f"{json_index-10}_{json_index}.{ext}")
        if not os.path.exists(last_json_path):
            logger.warning(f"{last_json_path}  not exists, only draw indices >0")
    else:
        last_json_path = None


    infos = load_dwpose_info(info_path)
    assert ori_vid_shape is not None
    new_shape = get_resize_shape_v2(ori_vid_shape, image_size)
    H,W,C = ori_vid_shape[:]
    render_frames = np.zeros((len(current_frame_indice),H,W,C))
    if info_class in ["rtmw_pose","sapiens_2b"]:
        use_rtmw = rtmw='rtmw' in info_class
        print(f"info_path:{info_path} \n info_class:{info_class} \n use_rtmw:{use_rtmw} ")
        if len(current_frame_indice)>0:
            render_frames = draw_dwpose_lmk_wild(render_frames,infos['dwpose'],current_frame_indice,rtmw=use_rtmw,draw_hand_color='ori')
        if last_json_path is not None:
            last_render_frames = np.zeros((len(last_frame_indice),H,W,C))
            print(f"last_json_path:{last_json_path}")
            last_info = load_dwpose_info(last_json_path)
            last_render_frames = draw_dwpose_lmk_wild(last_render_frames,last_info['dwpose'],last_frame_indice,rtmw=use_rtmw,draw_hand_color='ori')
            render_frames = np.concatenate([last_render_frames,render_frames])
    else:
        render_frames = draw_dwpose_lmk(render_frames, infos['dwpose'], current_frame_indice) # (N, H, W, C)
        if last_json_path is not None:
            last_render_frames = np.zeros((len(last_frame_indice),H,W,C))
            last_info = load_dwpose_info(last_json_path)
            last_render_frames = draw_dwpose_lmk(last_render_frames,last_info['dwpose'],last_frame_indice)
            render_frames = np.concatenate([last_render_frames,render_frames])
            



    
    transform_func_key = crop_kwargs.get("transform_function","transforms_videos")
    crop_func_key = crop_kwargs.get('crop_function','fix_crop_video')
    logger.info(f"transform_key:{transform_func_key} crop_key:{crop_func_key}")
    
    transforms_func = transform_func_dict[transform_func_key]
    render_frames = transforms_func(render_frames,new_shape)

    crop_func = crop_func_dict[crop_func_key]
    if 'crop_hook' in crop_kwargs.keys():
        render_frames = crop_kwargs['crop_hook'](render_frames)
    
    render_frames,_ = crop_func(render_frames,image_size,**(crop_kwargs.get('crop_function_kwargs',{})))
    # render_frames = fix_crop_video(render_frames,image_size,center_crop=center_crop)
    return render_frames

def read_test_data(video_path, frame_indice, image_size=768):
    # decord.bridge.set_bridge("torch")
    render_path = video_path.replace('/videos/', '/renders/')
    info_path = video_path.replace('/videos/', '/hamer_hifi/').replace('.mp4', '.pickle')
    infos = pickle.load(open(info_path, 'rb'))
    video = decord.VideoReader(video_path)
    render = decord.VideoReader(render_path)
    frame_indice.append(0)
    video_frames = video.get_batch(frame_indice).asnumpy() # (N, H, W, C)
    render_frames = render.get_batch(frame_indice).asnumpy() # (N, H, W, C)
    # render_frames = draw_dwpose_v2(render_frames, infos["dwpose"], frame_indice) # (N, H, W, C)
    render_frames = draw_dwpose_v4(render_frames, infos["dwpose"], frame_indice) # (N, H, W, C)
    new_shape = get_resize_shape_v2(video_frames[0].shape, image_size)
    video_frames = transforms_videos(video_frames, new_shape) # numpy(N, H, W, C) 0~255 -> torch(N, C, H, W) -1~1
    render_frames = transforms_videos(render_frames, new_shape)

    video_frames = keep_video_crop(video_frames, image_size)
    render_frames = keep_video_crop(render_frames, image_size)


    return video_frames, render_frames



def get_resize_shape_v2(shapes, max_size=512):
    # 保留最短边为max_size
    H, W, _ = shapes
    if H < W:
        resize_H = max_size
        # resize_W = int(resize_H / H * W) // 16 * 16
        resize_W = int(resize_H / H * W)
    else:
        resize_W = max_size
        resize_H = int(resize_W / W * H)
    return (resize_H, resize_W)

def get_resize_shape_hoi(shapes, max_size=512):
    # 保留最短边为max_size
    _, H, W = shapes
    if H < W:
        resize_H = max_size
        # resize_W = int(resize_H / H * W) // 16 * 16
        resize_W = int(resize_H / H * W)
    else:
        resize_W = max_size
        resize_H = int(resize_W / W * H)
    return (resize_H, resize_W)


def padding_video(video, max_size, green_bg=False):
    B, C, H, W = video.shape
    new_video = - torch.ones((len(video), 3, max_size, max_size))
    if H > W:
        resize_H = max_size
        resize_W = int(resize_H / H * W)
    else:
        resize_W = max_size
        resize_H = int(resize_W / W * H)
    video = torch.nn.functional.interpolate(video, size=(resize_H, resize_W), mode="bicubic", align_corners=True, antialias=True)
    if H > W:
        left = (max_size - resize_W) // 2
        new_video[:, :, :, left:left+resize_W] = video
    else:
        up = (max_size - resize_H) // 2
        new_video[:, :, up:up+resize_H] = video
    return new_video


def draw_dwpose_single(render_frame, dwpose):
    line_list = [[1, 2], [1, 5], [2, 3], [5, 6], [3, 4],  [6, 7]]
    colors = [[255, 0, 0],  [170, 255, 0], [0, 255, 170], [0, 255, 255], [0, 170, 255], [255, 0, 85]]
    candidate = np.array(dwpose)[:18]
    H, W, C = render_frame.shape
    # candidate = candidate * np.array([W, W]).reshape(1, 2)
    thick = 20 * min(H, W) // 1080
    zero_image = np.zeros_like(render_frame).astype(np.float32)
    for c, line in enumerate(line_list):
        Y = candidate[line, 0] * float(W)
        X = candidate[line, 1] * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), thick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(zero_image, polygon, colors[c])
    render_frame_mask = render_frame > 10
    render_frame = render_frame + (1 - render_frame_mask) * zero_image
    return render_frame


def get_resize_shape_v3(shapes, max_size=512):
    # 保留最短边为max_size
    H, W, _ = shapes
    if H < W:
        resize_H = max_size
        resize_W = int(resize_H / H * W)
    else:
        resize_W = max_size
        resize_H = int(resize_W / W * H)
    return (resize_H, resize_W)
    
def prepare_reference_input(video_path, image_size=768):
    video = decord.VideoReader(video_path)
    hand_render_path = video_path.replace('/videos/', '/smplx_mano/')
    face_render_path = video_path.replace('/videos/', '/face_renders/')
    info_path = video_path.replace('/videos/', '/smplx2d/').replace('.mp4', '.pickle')
    video = decord.VideoReader(video_path)
    hand_render = decord.VideoReader(hand_render_path)
    face_render = decord.VideoReader(face_render_path)
    video_frames = video.get_batch([0]).asnumpy()
    hand_render_frames = hand_render.get_batch([0]).asnumpy() # (N, H, W, C)
    face_render_frames = face_render.get_batch([0]).asnumpy() # (N, H, W, C)
    infos = pickle.load(open(info_path, 'rb'))["smplx2d"][0]

    render_frame = face_render_frames[0] + hand_render_frames[0]
    render_frame = draw_dwpose_single(render_frame, infos) # (N, H, W, C)
    new_shape = get_resize_shape_v3(video_frames[0].shape, image_size)
    video_frame = transforms_videos(video_frames, new_shape)[0] # numpy(N, H, W, C) 0~255 -> torch(N, C, H, W) -1~1
    render_frame = transforms_videos([render_frame], new_shape)[0]
    left, up = get_crop_box(video_frame.shape)

    video_frame = video_frame[:, up:up+image_size, left:left+image_size]
    render_frame = render_frame[:, up:up+image_size, left:left+image_size]
    return video_frame, render_frame

def get_crop_box(video_frame_shape, image_size=768,center_crop=True):
    C, H, W = video_frame_shape
    left = 0
    up = 0
    if H == image_size and W == image_size:
        left = 0
        up = 0
    elif H == image_size:
        left = (W - image_size - 1) // 2
    else:
        if center_crop:
            up = (H - image_size - 1) // 2
        else:
            up = 0
    return left, up

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     dataset = HumanDataset("", video_size=768, fps=25, max_num_frames=7, skip_frms_num=0)
#     loader = DataLoader(
#         dataset,
#         batch_size=1,
#         # sampler=sampler,
#         num_workers=0,
#         pin_memory=True,
#         # collate_fn=dataloader_collate_fn,
#     )
#     for i, item in enumerate(loader):
#         print(i)
#         print('video_frames:', item["video_frames"].shape)
#         print('render_frames:', item["render_frames"].shape)
#         print('num_frames:', item["num_frames"].shape)


def transforms_videos_hoi(video_frames, target_size, interpolation_mode="bicubic"):
    video_frames = torch.nn.functional.interpolate(video_frames, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)
    video_frames = (video_frames.float() / 255.0 - 0.5) / 0.5
    return video_frames

def read_hoi_data(data_root, data_dict,frame_indice,image_size):
    gt_path, pose_path, obj_mask_path, obj_path = \
                     data_dict['video_path'], data_dict['pose_path'], data_dict['obj_mask_path'], data_dict['obj_video_path']

    gt_path = os.path.join(data_root, gt_path)
    pose_path = os.path.join(data_root, pose_path)
    obj_mask_path = os.path.join(data_root, obj_mask_path)
    obj_path = os.path.join(data_root, obj_path)
    
    video = decord.VideoReader(gt_path)
    video_frames = video.get_batch(frame_indice).asnumpy() # (N, H, W, C) 
    H, W = video_frames.shape[1], video_frames.shape[2]
    width, height = W, H
    render_frames = np.zeros_like(video_frames)

    video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
    wo_obj_video_frames = video_frames.clone()

    ## ref frame generation
    obj_video = decord.VideoReader(obj_path)
    ref_idx = random.randint(0, len(obj_video) - 1)
    ref_img = obj_video[ref_idx].asnumpy() # (N, H, W, C)
    pixel_values_ref_img = torch.permute(torch.tensor(np.array(ref_img)), (2, 0, 1)).unsqueeze(0)


    ## Layout and mesh Pose and img inpaint
    obj_masks = decord.VideoReader(obj_mask_path)
    with open(pose_path, "r") as f:
        gt_2dpose = json.load(f)    # json keys: ['crop_video_path', 'box', 'width', 'height', 'dwpose']
    

    # import ipdb;ipdb.set_trace()
    rtmw = True
    draw_hand_color = "ori"
    render_frames = draw_dwpose_lmk_wild(render_frames, gt_2dpose['dwpose'], frame_indice, rtmw=rtmw, draw_hand_color=draw_hand_color, only_hand=True) # (N, H, W, C)
    render_frames = torch.permute(torch.tensor(np.array(render_frames)), (0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

    layout_path = gt_path.replace("clip_videos", "layout")
    hand_mask_path = gt_path.replace("clip_videos", "hand_mask")

    # import ipdb;ipdb.set_trace()
    vr_layout = decord.VideoReader(layout_path)
    vr_hand_mask = decord.VideoReader(hand_mask_path)

    pixel_values_box  = vr_layout.get_batch(frame_indice).asnumpy()
    hand_masks = vr_hand_mask.get_batch(frame_indice).asnumpy()

    pixel_values_box = torch.from_numpy(np.array(pixel_values_box)).permute(0, 3, 1, 2)
    hand_masks = torch.from_numpy(np.array(hand_masks)).permute(0, 3, 1, 2)/255.0
    
    pixel_values_mesh = render_frames*hand_masks
    

    
    for i,idx in enumerate(frame_indice):
        img_mask = obj_masks[idx].asnumpy()
        img_mask = np.array(img_mask) / 255
        mask_idx = np.where(img_mask == 1)

        # if mask_idx==None:
        #     continue
        try:
            h1,h2,w1,w2 = min(mask_idx[0]),max(mask_idx[0]),min(mask_idx[1]),max(mask_idx[1])
            h = h2-h1
            w = w2-w1
            center_h = (h1+h2)//2
            center_w = (w1+w2)//2
            mask_size = int((max(h,w)*1.25)//2 * 2)
            # import ipdb;ipdb.set_trace()

            wo_obj_video_frames[i,:,center_h-(mask_size//2):center_h+(mask_size//2),center_w-(mask_size//2):center_w+(mask_size//2)] = 0
        except:
            pass

    hand_masks = hand_masks*255.
    # import ipdb;ipdb.set_trace()
    new_shape = get_resize_shape_hoi(video_frames[0].shape, image_size) #torch.Size([126, 3, 557, 512]) 69 70  8/2
    new_shape = (round(new_shape[0]/16)*16, round(new_shape[1]/16)*16)
    wo_obj_video_frames = transforms_videos_hoi(wo_obj_video_frames, new_shape)
    pixel_values_mesh   = transforms_videos_hoi(pixel_values_mesh, new_shape)
    pixel_values_box    = transforms_videos_hoi(pixel_values_box, new_shape)
    hand_masks = transforms_videos_hoi(hand_masks, new_shape)

    video_frames = torch.cat([pixel_values_ref_img, video_frames], dim=0)
    video_frames = transforms_videos_hoi(video_frames, new_shape) # numpy(N, H, W, C) 0~255 -> torch(N, C, H, W) -1~1

    return video_frames, wo_obj_video_frames, pixel_values_mesh, pixel_values_box, hand_masks



def test_dataset():
    from torch.utils.data import DataLoader
    # 修改 data_dir 为你的视频列表文件路径，若为空则默认使用 "data/training_0801.list"
    data_dir = "/root/paddlejob/workspace/shenzhelun/wan/shenzhelun/train_hoi_data.csv"  
    try:
        # 实例化数据集
        dataset = HumanHoiDataset(data_dir=data_dir,
                                   video_size=(832,480),
                                   fps=25,
                                   max_num_frames=81,
                                   skip_frms_num=3,
                                   ref_id_type="random",
                                   draw_hand_color="ori",
                                   info_class="dwpose_test123")
        
        print("数据集长度:", len(dataset))
        
        # 使用 DataLoader 测试迭代
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        
        # 取出一个 batch 进行测试
        for i, sample in enumerate(dataloader):
            print("样本", i)
            # 检查视频帧和渲染帧的形状
            # video_frames = sample["video_frames"]
            # render_frames = sample["render_frames"]
            num_frames = sample["num_frames"]
            fps = sample["fps"]
            prompt = sample["prompt"]
            
            # print("video_frames shape:", video_frames.shape)
            # print("render_frames shape:", render_frames.shape)
            print("prompt:", prompt)
            print("num_frames:", num_frames)
            print("fps:", fps)
            # 仅测试一个 batch
            break

    except Exception as e:
        print("测试数据集时出现异常:", e)
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()