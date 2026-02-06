import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import decord
from PIL import Image
import json


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


def resize_with_padding(input_tensor, target_h, target_w, additional_pixels=0, bg=0):
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
        pad_w // 2 + additional_pixels,          # 左
        pad_w - pad_w // 2 + additional_pixels,  # 右
        pad_h // 2 + additional_pixels,          # 上
        pad_h - pad_h // 2 + additional_pixels   # 下
    )

    # 应用填充（填充值为0）
    padded = F.pad(resized, padding, mode="constant", value=bg)

    return padded


def get_frame_range(vr, frames, sample_rate=1, start_end=None, begin_at_first_frame=False):
    max_range = len(vr)
    min_range = 0
    if start_end is not None:
        max_range = start_end[1]
        min_range = start_end[0]
    if max_range - sample_rate * frames - 1 < 0:
        frames = max_range

    frame_start = max_range - sample_rate * frames - 1
    if frame_start < min_range:
        frame_start = min_range

    if begin_at_first_frame:
        frame_number_start = 0
    else:
        frame_number_start = random.randint(min_range, frame_start)
    frame_range = range(frame_number_start, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:frames]
    return frame_range_indices


def random_crop_videos_multi_res(video_frames, render_frames=None, target_size=None):
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


def apply_mask_and_crop(img_path, mask_path):
    # 1. 读取图像和 mask
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    if img.size != mask.size:  # 注意：PIL 的 size 是 (width, height)
        img = img.resize(mask.size, resample=Image.BILINEAR)
        
    # 2. 转为 numpy 数组
    img_np = np.array(img)           # (H, W, 3)
    mask_np = np.array(mask)         # (H, W), uint8

    # 3. 创建布尔掩膜：非零为有效区域
    bool_mask = mask_np > 0          # dtype: bool, shape: (H, W)
    bool_mask_3c = np.stack([bool_mask]*3, axis=-1)  # (H, W, 3)

    # 4. 将无效区域设为黑色
    img_np[~bool_mask_3c] = 0

    # 5. 计算有效区域的边界框
    ys, xs = np.where(bool_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask 中没有有效区域，无法裁剪")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 6. 裁剪图像和 mask
    cropped_img_np = img_np[y_min:y_max+1, x_min:x_max+1, :]
    cropped_mask_np = mask_np[y_min:y_max+1, x_min:x_max+1]

    return cropped_img_np, cropped_mask_np


class HumanHoiDataset_anchorcrafter(Dataset):
    def __init__(self, data_dir="", video_size=768, fps=25, max_num_frames=7, skip_frms_num=3, ref_id_type="random", draw_hand_color="ori", info_class="dwpose_test123", data_aug=True, is_random=True, ref_img=None, ref_first_frame=False, is_test=False, is_rehold=True, is_fl=False):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(HumanHoiDataset_anchorcrafter, self).__init__()
        
        self.video_size = video_size  # [768, 768]
        self.fps = fps
        self.max_num_frames = max_num_frames  # 33
        self.skip_frms_num = skip_frms_num    # 3.0
        self.aug = data_aug
        self.random = is_random
        self.is_test = is_test
        self.is_rehold = is_rehold
        self.is_fl = is_fl

        if data_dir is None or data_dir == "":
            video_list_path = "data/training_0801.list"
        else:
            video_list_path = data_dir

        ### load video list
        with open(video_list_path, 'r') as csvfile:
            import csv
            self.videos_list = list(csv.DictReader(csvfile))

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
        self.caption_path = "prompt/prompt_all_v2.txt"
        self.caption_root = "/root/paddlejob/workspace/huangxuan/bos_data/yqw/processed_human_videos"
        
        # Check if caption file exists
        if os.path.exists(self.caption_path):
            with open(self.caption_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.caption = json.loads(content)
        else:
            print(f"Warning: Caption file not found at {self.caption_path}, caption feature disabled.")
            self.caption = None
        self.ref_image = ref_img
        self.ref_first_frame = ref_first_frame
        print('ref_id_type:', ref_id_type)
        print("frame_number_list:", frame_number_list)
        print(f"draw_hand_color:{draw_hand_color}")

    def __getitem__(self, index):
        video_index = index
        if "\t" in self.videos_list[video_index]:
            video_path, start_end = self.videos_list[video_index].split("\t")
            start, end = start_end.split(":")
            start_end = [int(start), int(end)]
        else:
            data_dict = self.videos_list[video_index]
            start_end = None
        
        gt_path, pose_path, obj_mask_path, obj_path, ref_mask_path = \
            data_dict['video_path'], data_dict['video_path'], data_dict['obj_mask_path'], data_dict['obj_path'], data_dict['mask_path'],

        gt_path = os.path.join(self.data_root, gt_path)
        pose_path = os.path.join(self.data_root, pose_path)
        obj_mask_path = os.path.join(self.data_root, obj_mask_path)
        obj_path = os.path.join(self.data_root, obj_path) + "/01.jpg"
        ref_mask_path = os.path.join(self.data_root, ref_mask_path) + "/01.jpg"
        txt_path = gt_path.replace(".mp4", ".txt")
        prompt_key = gt_path.replace(self.data_root, self.caption_root)
        if self.ref_first_frame:
            prompt = ""
        else:
            if not self.is_rehold and self.caption is not None and prompt_key in self.caption:
                prompt = self.caption[prompt_key]
            else:
                prompt = ""

        video = decord.VideoReader(gt_path)
        fps = int(video.get_avg_fps())
        if start_end is not None:
            start_end = [int(x * fps) for x in start_end]
        
        size_index = np.random.choice(range(len(self.image_size_list)), p=self.probs)
        resolution_list = get_all_resolution(self.image_size_list[size_index])
        frame_num = self.max_num_frames
        cur_resolution = self.video_size
        sample_rate = 1

        ## motion frames + gt frames
        frame_indice = get_frame_range(video, frames=frame_num, sample_rate=sample_rate, start_end=start_end, begin_at_first_frame=self.ref_first_frame)

        video_frames = video.get_batch(frame_indice).asnumpy()  # (N, H, W, C)
        render_frames = np.zeros_like(video_frames)
        H, W = video_frames.shape[1], video_frames.shape[2]
        width, height = W, H

        video_frames = torch.permute(torch.tensor(np.array(video_frames)), (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        wo_obj_video_frames = video_frames.clone()

        ## ref frame generation
        if not self.ref_image:
            padding_scale = random.randint(0, 40)
            ref_img, cropedmask = apply_mask_and_crop(obj_path, ref_mask_path)
            pixel_values_ref_img_o = torch.permute(torch.tensor(ref_img), (2, 0, 1)).unsqueeze(0)
            pixel_values_ref_img = resize_with_padding(pixel_values_ref_img_o, target_h=video_frames.shape[2], target_w=video_frames.shape[3])

            mask = (pixel_values_ref_img_o[:, 0] == 0) & \
                (pixel_values_ref_img_o[:, 1] == 0) & \
                (pixel_values_ref_img_o[:, 2] == 0)
            pixel_values_ref_img_o[:, 0][mask] = 128
            pixel_values_ref_img_o[:, 1][mask] = 128
            pixel_values_ref_img_o[:, 2][mask] = 128

            mask = (pixel_values_ref_img[:, 0] == 0) & \
                (pixel_values_ref_img[:, 1] == 0) & \
                (pixel_values_ref_img[:, 2] == 0)
            pixel_values_ref_img[:, 0][mask] = 128
            pixel_values_ref_img[:, 1][mask] = 128
            pixel_values_ref_img[:, 2][mask] = 128

        else:
            img = Image.open(self.ref_image).convert('RGB')
            ref_img = np.array(img)
            pixel_values_ref_img_o = torch.permute(torch.tensor(ref_img), (2, 0, 1)).unsqueeze(0)
            pixel_values_ref_img_o = resize_with_padding(pixel_values_ref_img_o, 384, 384, additional_pixels=60)

            mask = (pixel_values_ref_img_o[:, 0] == 0) & \
                (pixel_values_ref_img_o[:, 1] == 0) & \
                (pixel_values_ref_img_o[:, 2] == 0)
            pixel_values_ref_img_o[:, 0][mask] = 128
            pixel_values_ref_img_o[:, 1][mask] = 128
            pixel_values_ref_img_o[:, 2][mask] = 128
            pixel_values_ref_img = resize_with_padding(pixel_values_ref_img_o, target_h=video_frames.shape[2], target_w=video_frames.shape[3])

        if random.random() < 0.5 and self.aug:
            pixel_values_ref_img = apply_color_augmentation(
                pixel_values_ref_img,
                brightness=0.2, brightness_prob=0.5,
                contrast=0.2, contrast_prob=0.5,
                saturation=0.2, saturation_prob=0.5,
                hue=0.05, hue_prob=0.5
            )

        pixel_values_ref_img = pixel_values_ref_img.repeat(self.max_num_frames, 1, 1, 1)
        pixel_values_ref_img_0 = torch.zeros_like(pixel_values_ref_img)
        pixel_values_ref_img_only_mask = pixel_values_ref_img_0

        ## Layout and mesh Pose and img inpaint
        obj_masks = decord.VideoReader(obj_mask_path)

        if random.random() < 0.1 and self.random:
            random_box = True
        else:
            random_box = False
        
        mask_size = -1
        if self.is_test:
            mask_scale = 1.1
        else:
            mask_scale = round(random.uniform(1.05, 1.35), 2)
        for i, idx in enumerate(frame_indice):
            img_mask = obj_masks[idx].asnumpy()
            img_mask = np.array(img_mask) / 255
            mask_idx = np.where(img_mask > 0.5)
            try:
                h1, h2, w1, w2 = min(mask_idx[0]), max(mask_idx[0]), min(mask_idx[1]), max(mask_idx[1])
                h = h2 - h1
                w = w2 - w1
                mask_size_frame = int((max(h, w) * mask_scale) // 2 * 2)
                mask_size = max(mask_size, mask_size_frame)
            except:
                pass

        for i, idx in enumerate(frame_indice):
            if i == 0:
                continue
            if self.is_fl and (i % 80 == 0):
                continue
            try:
                img_mask = obj_masks[idx].asnumpy()
                img_mask = np.array(img_mask) / 255
                mask_idx = np.where(img_mask > 0.5)

                h1, h2, w1, w2 = min(mask_idx[0]), max(mask_idx[0]), min(mask_idx[1]), max(mask_idx[1])
                h = h2 - h1
                w = w2 - w1
                if random_box:
                    center_h = (h1 + h2) // 2 + random.randint(-10, 10)
                    center_w = (w1 + w2) // 2 + random.randint(-10, 10)
                else:
                    center_h = (h1 + h2) // 2
                    center_w = (w1 + w2) // 2
                half = mask_size // 2
                img_h, img_w = wo_obj_video_frames.shape[-2], wo_obj_video_frames.shape[-1]
                center_h = min(max(center_h, half), img_h - half)
                center_w = min(max(center_w, half), img_w - half)
                pixel_values_ref_img_mask = resize_with_padding(pixel_values_ref_img_o, target_h=mask_size, target_w=mask_size, bg=128)
                wo_obj_video_frames[i, :, center_h-(mask_size//2):center_h+(mask_size//2), center_w-(mask_size//2):center_w+(mask_size//2)] = 128
                pixel_values_ref_img_only_mask[i, :, center_h-(mask_size//2):center_h+(mask_size//2), center_w-(mask_size//2):center_w+(mask_size//2)] = 255
            except:
                pass
                print("error!!!", gt_path, pose_path, obj_mask_path, obj_path, ref_mask_path)

        ### 这是一个临时方案，由于代码里面没有实际使用这几个condition
        pixel_values_mesh = video_frames
        pixel_values_box = video_frames
        hand_masks = video_frames

        video_frames = torch.cat([video_frames, wo_obj_video_frames, pixel_values_ref_img, pixel_values_ref_img_only_mask], dim=0)
        render_frames = torch.cat([pixel_values_mesh, pixel_values_box, hand_masks], dim=0)

        video_frames, render_frames = random_crop_videos_multi_res(video_frames, render_frames, cur_resolution)

        gt_frames = video_frames[:self.max_num_frames, :, :, :]
        wo_obj_video_frames = video_frames[self.max_num_frames:self.max_num_frames * 2, :, :, :]
        pixel_values_ref_img = video_frames[self.max_num_frames * 2:self.max_num_frames * 3, :, :, :]
        pixel_values_ref_img_only_mask = video_frames[self.max_num_frames * 3:self.max_num_frames * 4, :, :, :]
        hand_pose = render_frames[:self.max_num_frames, :, :, :]
        hand_obj_box = render_frames[self.max_num_frames:self.max_num_frames * 2, :, :, :]

        item = {
            "gt_frames": gt_frames.contiguous(),
            "wo_obj_video_frames": wo_obj_video_frames.contiguous(),
            "pixel_values_ref_img": pixel_values_ref_img[0:1].contiguous(),
            "hand_pose": hand_pose.contiguous(),
            "hand_obj_box": hand_obj_box.contiguous(),
            "pixel_values_ref_img_only_mask": pixel_values_ref_img_only_mask.contiguous(),
            "num_frames": frame_num,
            "prompt": prompt,
            "data_path": gt_path,
            "fps": 25 // sample_rate,
        }
        return item

    def __len__(self):
        return len(self.videos_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)