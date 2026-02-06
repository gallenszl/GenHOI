import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import decord
from decord import VideoReader
from torch.utils.data import Dataset
import json
from PIL import Image
import traceback
import csv
from torchvision.transforms.functional import to_pil_image


def tensor_to_pil_list(tensor):
    """
    Convert a tensor of shape [T, C, H, W] to list of PIL.Image
    Supports input in ranges:
        [-1, 1]  -> normalized to [0, 255]
        [0, 1]   -> scaled to [0, 255]
        [0, 255] -> converted directly
    """
    imgs = []
    tensor = tensor.detach().cpu()

    for frame in tensor:
        frame_min, frame_max = frame.min().item(), frame.max().item()

        if frame_min >= -1.0 and frame_max <= 1.0:
            if frame_min < 0:  # assume [-1, 1]
                frame = (frame + 1) / 2  # [-1, 1] -> [0, 1]
            frame = frame * 255.0  # [0, 1] -> [0, 255]

        frame = frame.clamp(0, 255).byte()
        imgs.append(to_pil_image(frame))

    return imgs


def resize_with_padding(input_tensor, target_h, target_w, additional_pixels=0):
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
        mode="nearest"
    )

    # 计算需要填充的像素数
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    # 对称填充（上下、左右均分填充量）
    padding = (
        pad_w // 2 + additional_pixels,          # 左
        pad_w - pad_w // 2 + additional_pixels,  # 右
        pad_h // 2 + additional_pixels,          # 上
        pad_h - pad_h // 2 + additional_pixels   # 下
    )

    # 应用填充（填充值为0）
    padded = F.pad(resized, padding, mode="constant", value=0)

    return padded


def get_frame_range(vr, frames, sample_rate=1, start_end=None, begin_at_first_frame=False):
    max_range = len(vr)
    min_range = 0

    frame_start = 0
        
    if begin_at_first_frame:
        frame_number_start = 0
    else:
        frame_number_start = random.randint(min_range, frame_start)
    frame_range = range(frame_number_start, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:frames]
    return frame_range_indices


def get_all_resolution(image_size):
    all_resolution = []
    divided_by = 32
    min_edge = int(image_size / 1.4)
    max_edge = int(image_size * 1.4)
    token_number = image_size * image_size / divided_by / divided_by
    for i in range(min_edge // divided_by, max_edge // divided_by + 1):
        all_resolution.append([i * divided_by, int(token_number // i * divided_by)])
    return all_resolution


def random_crop_videos_multi_res(
    video_frames: torch.Tensor,
    render_frames: torch.Tensor = None,
    target_size=None,
    return_offset: bool = False,
    protect_bbox=None,
    pad_px: int = 0
):
    """
    随机裁剪 -> 缩放到 target_size
    """
    import math
    F_, C, H, W = video_frames.shape
    th, tw = target_size
    target_ratio = th / float(tw)
    origin_ratio = H / float(W)

    if target_ratio > origin_ratio:
        crop_h = H if H <= th else random.randint(int(th), H - 1)
        crop_w = int(crop_h / target_ratio)
        top  = 0 if H <= th else random.randint((H - crop_h) // 7, (H - crop_h) // 7 * 3)
        left = 0 if W <= crop_w else random.randint((W - crop_w - 1)//7*3, (W - crop_w - 1)//7*4)
    else:
        crop_w = W if W <= tw else random.randint(int(tw), W - 1)
        crop_h = int(crop_w * target_ratio)
        left = 0 if W <= tw else random.randint((W - crop_w - 1)//7*3, (W - crop_w - 1)//7*4)
        top  = 0 if H <= crop_h else random.randint((H - crop_h - 1)//7, (H - crop_h - 1)//7 * 3)

    if protect_bbox is not None:
        t0, l0, b0, r0 = protect_bbox
        l0 = max(0, l0 - pad_px); t0 = max(0, t0 - pad_px)
        r0 = min(W, r0 + pad_px); b0 = min(H, b0 + pad_px)
        box_w = max(1, r0 - l0)
        box_h = max(1, b0 - t0)

        if (box_h / float(box_w)) >= target_ratio:
            min_h = box_h
            min_w = int(math.ceil(min_h / target_ratio))
        else:
            min_w = box_w
            min_h = int(math.ceil(min_w * target_ratio))

        if crop_h < min_h or crop_w < min_w:
            crop_h = max(crop_h, min_h)
            crop_w = max(crop_w, min_w)

            if crop_h > H:
                crop_h = H
                crop_w = int(round(crop_h / target_ratio))
            if crop_w > W:
                crop_w = W
                crop_h = int(round(crop_w * target_ratio))

        left_min = max(0, r0 - crop_w);     left_max = min(l0, W - crop_w)
        top_min  = max(0, b0 - crop_h);     top_max  = min(t0, H - crop_h)

        if left_min <= left_max:
            left = max(left_min, min(left, left_max))
        else:
            left = int(round((l0 + r0 - crop_w) / 2.0))
            left = max(0, min(W - crop_w, left))

        if top_min <= top_max:
            top = max(top_min, min(top, top_max))
        else:
            top = int(round((t0 + b0 - crop_h) / 2.0))
            top = max(0, min(H - crop_h, top))

        if left > l0: left = l0
        if top  > t0: top  = t0
        if left + crop_w < r0: left = r0 - crop_w
        if top  + crop_h < b0: top  = b0 - crop_h
        left = max(0, min(W - crop_w, left))
        top  = max(0, min(H - crop_h, top))

    v = video_frames[:, :, top:top+crop_h, left:left+crop_w]
    r = render_frames[:, :, top:top+crop_h, left:left+crop_w] if render_frames is not None else None

    v = torch.nn.functional.interpolate(v, size=(th, tw), mode="bilinear", align_corners=True, antialias=True)
    if r is not None:
        r = torch.nn.functional.interpolate(r, size=(th, tw), mode="bilinear", align_corners=True, antialias=True)

    v = (v.float() / 255.0 - 0.5) / 0.5
    if r is not None:
        r = (r.float() / 255.0 - 0.5) / 0.5

    if return_offset:
        return v, r, (top, left, crop_h, crop_w)
    return v, r


def center_crop_videos_multi_res(video_frames, render_frames=None, target_size=None, return_offset=False):
    T, C, H, W = video_frames.shape
    th, tw = target_size
    target_ratio = th / tw
    origin_ratio = H / W

    if origin_ratio > target_ratio:
        crop_h = int(W * target_ratio)
        crop_w = W
    else:
        crop_h = H
        crop_w = int(H / target_ratio)

    top  = (H - crop_h) // 2
    left = (W - crop_w) // 2

    v = video_frames[:, :, top:top+crop_h, left:left+crop_w]
    r = render_frames[:, :, top:top+crop_h, left:left+crop_w] if render_frames is not None else None

    v = torch.nn.functional.interpolate(v, size=(th, tw), mode="bilinear", align_corners=True, antialias=True)
    if r is not None:
        r = torch.nn.functional.interpolate(r, size=(th, tw), mode="bilinear", align_corners=True, antialias=True)

    v = (v.float() / 255.0 - 0.5) / 0.5
    if r is not None:
        r = (r.float() / 255.0 - 0.5) / 0.5

    if return_offset:
        return v, r, (top, left, crop_h, crop_w)
    return v, r


class HumanHoiDataset_inference(Dataset):
    def __init__(self, data_dir="", video_size=None, fps=25, max_num_frames=None, skip_frms_num=3,
                 ref_id_type="random", draw_hand_color="ori", info_class="dwpose_test123", 
                 data_aug=True, is_random=True, ref_img=None, ref_first_frame=False, 
                 is_test=False, is_stage1=False, scale=1.25, is_fl=False, ref_in_bbox=True, 
                 ref_bg=128, data_root=None, ref_first_bbox=False, ref_first_img=False):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(HumanHoiDataset_inference, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.aug = data_aug
        self.random = is_random
        self.is_test = is_test
        self.is_stage1 = is_stage1
        self.scale = scale
        self.is_fl = is_fl
        self.ref_bg = ref_bg
        self.ref_first_bbox = ref_first_bbox
        self.ref_first_img = ref_first_img

        if data_dir is None or data_dir == "":
            video_list_path = "data/training_0801.list"
        else:
            video_list_path = data_dir

        ### load video list
        with open(video_list_path, 'r') as csvfile:
            self.videos_list = list(csv.DictReader(csvfile))
            if not self.is_test:
                random.shuffle(self.videos_list)

        image_size_list = [x * 8 for x in range(96, 97, 2)]
        image_size_list = [512]
        frame_number_list = [int(768. * 768. / x / x * 17) for x in image_size_list]
        for i in range(len(frame_number_list)):
            frame_number_list[i] = 33 + 5
        probs = [x+1 for x in range(len(frame_number_list))]
        self.probs = [float(x) / sum(probs) for x in probs]
        self.frame_number_list = frame_number_list
        self.image_size_list = image_size_list
        self.ref_id_type = ref_id_type
        self.draw_hand_color = draw_hand_color
        self.info_class = info_class

        self.data_root = ""
        if data_root is not None:
            self.data_root = data_root
        self.caption_path = "/root/paddlejob/workspace/huangxuan/DiffSynth-Studio-New/prompt/prompt_all_v2.txt"
        self.caption_root = "/home/vis/bosdata/yqw/processed_human_videos"
        
        # 检测 caption 文件是否存在
        if os.path.exists(self.caption_path):
            with open(self.caption_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.caption = json.loads(content)
        else:
            print(f"Warning: Caption file not found at {self.caption_path}, caption feature disabled.")
            self.caption = None

        self.ref_image = ref_img
        self.ref_first_frame = ref_first_frame
        self.ref_in_bbox = ref_in_bbox
        print('ref_id_type:', ref_id_type)
        print("frame_number_list:", frame_number_list)
        print(f"draw_hand_color:{draw_hand_color}")


    def __getitem__(self, index):
        while True:
            if self.is_test:
                video_index = index
            else:
                video_index = random.randint(0, len(self.videos_list) - 1)
            if "\t" in self.videos_list[video_index]:
                video_path, start_end = self.videos_list[video_index].split("\t")
                start, end = start_end.split(":")
                start_end = [int(start), int(end)]
            else:
                data_dict = self.videos_list[video_index]
                start_end = None
            
            gt_path, obj_mask_path, input_path = \
                    data_dict['video_path'], data_dict['obj_mask_path'], data_dict['input_path']

            gt_path = os.path.join(self.data_root, gt_path)
            obj_mask_path = os.path.join(self.data_root, obj_mask_path)
            input_path = os.path.join(self.data_root, input_path)
            txt_path = gt_path.replace(".mp4", ".txt")
            prompt_key = gt_path.replace(self.data_root, self.caption_root)
            
            if self.ref_first_frame:
                prompt = ""
            else:
                if self.caption is not None and prompt_key in self.caption:
                    prompt = self.caption[prompt_key]
                else:
                    prompt = "" 
            if random.random() < 0.5:
                prompt = ""

            video = decord.VideoReader(gt_path)
            fps = int(video.get_avg_fps())
            if start_end is not None:
                start_end = [int(x * fps) for x in start_end]
            
            size_index = np.random.choice(range(len(self.image_size_list)), p=self.probs)
            resolution_list = get_all_resolution(self.image_size_list[size_index])
            if self.max_num_frames is None:
                frame_num = len(video)
            else:
                frame_num = self.max_num_frames
            sample_rate = 1

            ## motion frames + gt frames
            frame_indice = get_frame_range(video, frames=frame_num, sample_rate=sample_rate, 
                                           start_end=start_end, begin_at_first_frame=self.ref_first_frame)
            
            if len(frame_indice) != frame_num:
                if self.is_test:
                    print(f'{gt_path} not enough frames, but we use {len(frame_indice)} anyway.')
                else:
                    print(f'{gt_path} not enough frames')
                    continue

            video_frames = video.get_batch(frame_indice).asnumpy()
            H, W = video_frames.shape[1], video_frames.shape[2]
            width, height = W, H
            if self.video_size is None:
                cur_resolution = [H, W]
            else:
                cur_resolution = self.video_size
            video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2))

            wo_obj_video_frames = decord.VideoReader(input_path)
            wo_obj_video_frames = wo_obj_video_frames.get_batch(frame_indice).asnumpy()
            wo_obj_video_frames = torch.permute(torch.tensor(np.array(wo_obj_video_frames)), (0, 3, 1, 2))

            obj_mask = decord.VideoReader(obj_mask_path)
            pixel_values_ref_img_only_mask = obj_mask.get_batch(frame_indice).asnumpy()
            pixel_values_ref_img_only_mask = torch.permute(torch.tensor(np.array(pixel_values_ref_img_only_mask)), (0, 3, 1, 2))

            first_mask_for_bbox = pixel_values_ref_img_only_mask[1, 0].clone()

            if self.is_fl:
                for i in range(0, frame_num, 80):
                    pixel_values_ref_img_only_mask[i] = 0
            else:
                pixel_values_ref_img_only_mask[0] = 0

            if not self.ref_in_bbox:
                mask = (pixel_values_ref_img_only_mask[:, 0] >= 10) & \
                       (pixel_values_ref_img_only_mask[:, 1] >= 10) & \
                       (pixel_values_ref_img_only_mask[:, 2] >= 10)
                wo_obj_video_frames[:, 0][mask] = 128
                wo_obj_video_frames[:, 1][mask] = 128
                wo_obj_video_frames[:, 2][mask] = 128

            ## ref frame generation
            img_path = data_dict['ref_img']
            img_path = os.path.join(self.data_root, img_path)
            img = Image.open(img_path).convert('RGB')
            ref_img = np.array(img)
            pixel_values_ref_img_o = torch.permute(torch.tensor(ref_img), (2, 0, 1)).unsqueeze(0)

            pixel_values_ref_img = resize_with_padding(pixel_values_ref_img_o, target_h=video_frames.shape[2], target_w=video_frames.shape[3])
            pixel_values_ref_img_ori_size = pixel_values_ref_img

            # ref_first_bbox: 从第一帧提取放大区域
            if self.ref_first_bbox:
                ys, xs = torch.where(first_mask_for_bbox > 128)
                if ys.numel() > 0 and xs.numel() > 0:
                    top, bottom = int(ys.min().item()), int(ys.max().item()) + 1
                    left, right = int(xs.min().item()), int(xs.max().item()) + 1

                    patch = wo_obj_video_frames[0:1, :, top:bottom, left:right]
                    H_, W_ = wo_obj_video_frames.shape[-2], wo_obj_video_frames.shape[-1]
                    _, C, h, w = patch.shape
                    scale = min(H_ / h, W_ / w)

                    new_h = max(int(h * scale), 1)
                    new_w = max(int(w * scale), 1)

                    patch_resized = torch.nn.functional.interpolate(
                        patch, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )

                    patch_full = torch.full_like(wo_obj_video_frames[0:1], fill_value=self.ref_bg)
                    top_pad = (H_ - new_h) // 2
                    left_pad = (W_ - new_w) // 2
                    patch_full[:, :, top_pad:top_pad+new_h, left_pad:left_pad+new_w] = patch_resized

                    pixel_values_ref_img_ori_size = torch.cat(
                        [pixel_values_ref_img_ori_size, patch_full], dim=0
                    )

            # 处理黑色和白色像素
            mask_black = (pixel_values_ref_img_o[:, 0] == 0) & \
                        (pixel_values_ref_img_o[:, 1] == 0) & \
                        (pixel_values_ref_img_o[:, 2] == 0)

            mask_white = (pixel_values_ref_img_o[:, 0] == 255) & \
                        (pixel_values_ref_img_o[:, 1] == 255) & \
                        (pixel_values_ref_img_o[:, 2] == 255)

            mask = mask_black | mask_white
            pixel_values_ref_img_o[:, 0][mask] = 128
            pixel_values_ref_img_o[:, 1][mask] = 128
            pixel_values_ref_img_o[:, 2][mask] = 128

            mask_black = (pixel_values_ref_img_ori_size[:, 0] == 0) & \
                        (pixel_values_ref_img_ori_size[:, 1] == 0) & \
                        (pixel_values_ref_img_ori_size[:, 2] == 0)

            mask_white = (pixel_values_ref_img_ori_size[:, 0] == 255) & \
                        (pixel_values_ref_img_ori_size[:, 1] == 255) & \
                        (pixel_values_ref_img_ori_size[:, 2] == 255)

            mask = mask_black | mask_white
            pixel_values_ref_img_ori_size[:, 0][mask] = self.ref_bg
            pixel_values_ref_img_ori_size[:, 1][mask] = self.ref_bg
            pixel_values_ref_img_ori_size[:, 2][mask] = self.ref_bg

            tmp_single_num = video_frames.shape[0]
            if self.scale != 1:
                mask_size = -1
                mask_scale = round(random.uniform(1.05, 1.35), 2)
                if self.is_test:
                    mask_scale = self.scale
                for i, idx in enumerate(frame_indice):
                    img_mask = obj_mask[idx].asnumpy()
                    img_mask = np.array(img_mask) / 255
                    mask_idx = np.where(img_mask == 1)

                    if mask_idx[0].shape[0] == 0:
                        continue

                    h1, h2, w1, w2 = min(mask_idx[0]), max(mask_idx[0]), min(mask_idx[1]), max(mask_idx[1])
                    h = h2 - h1
                    w = w2 - w1
                    mask_size_frame = int((max(h, w) * mask_scale) // 2 * 2)
                    mask_size = max(mask_size, mask_size_frame)

                for i in range(tmp_single_num):
                    img_mask = obj_mask[idx].asnumpy()
                    img_mask = np.array(img_mask) / 255
                    mask_idx = np.where(img_mask == 1)

                    if mask_idx[0].shape[0] == 0:
                        continue

                    single = pixel_values_ref_img_only_mask[i, 0]
                    ys, xs = torch.where(single > 128)
                    h1, h2 = int(ys.min().item()), int(ys.max().item())
                    w1, w2 = int(xs.min().item()), int(xs.max().item())
                    center_h = (h1 + h2) // 2
                    center_w = (w1 + w2) // 2

                    half = mask_size // 2
                    img_h, img_w = wo_obj_video_frames.shape[-2], wo_obj_video_frames.shape[-1]
                    center_h = min(max(center_h, half), img_h - half)
                    center_w = min(max(center_w, half), img_w - half)
                    pixel_values_ref_img_mask = resize_with_padding(pixel_values_ref_img_o, target_h=mask_size, target_w=mask_size)
                    wo_obj_video_frames[i, :, center_h-(mask_size//2):center_h+(mask_size//2), center_w-(mask_size//2):center_w+(mask_size//2)] = pixel_values_ref_img_mask
                    pixel_values_ref_img_only_mask[i, :, center_h-(mask_size//2):center_h+(mask_size//2), center_w-(mask_size//2):center_w+(mask_size//2)] = 255

            video_frames = torch.cat([video_frames, wo_obj_video_frames, pixel_values_ref_img_only_mask, pixel_values_ref_img_ori_size], dim=0)
            render_frames = None
            if self.is_test:
                video_frames, render_frames = center_crop_videos_multi_res(video_frames, render_frames, cur_resolution)
            else:    
                video_frames, render_frames = random_crop_videos_multi_res(video_frames, render_frames, cur_resolution)

            gt_frames = video_frames[:tmp_single_num, :, :, :]
            wo_obj_video_frames = video_frames[tmp_single_num:tmp_single_num * 2, :, :, :]
            pixel_values_ref_img_only_mask = video_frames[tmp_single_num * 2:tmp_single_num * 3, :, :, :]
            pixel_values_ref_img_ori_size = video_frames[tmp_single_num * 3:, :, :, :]

            if self.ref_first_img:
                pixel_values_ref_img_ori_size = torch.cat(
                    [wo_obj_video_frames[0:1], pixel_values_ref_img_ori_size], dim=0
                )

            item = {
                "video": tensor_to_pil_list(gt_frames.contiguous()),
                "vace_video": tensor_to_pil_list(wo_obj_video_frames.contiguous()),
                "vace_video_mask": tensor_to_pil_list(pixel_values_ref_img_only_mask.contiguous()),
                "vace_reference_image": tensor_to_pil_list(pixel_values_ref_img_ori_size.contiguous()),
                "prompt": prompt,
            }
            
            return item

    def __len__(self):
        return len(self.videos_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)