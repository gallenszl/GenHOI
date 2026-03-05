import os
import argparse
import random
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch.multiprocessing as mp
from diffsynth.models.set_condition_branch import set_stand_in, set_gate
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from examples.wanvideo.dataset.customer_dataset import HumanHoiDataset_inference
# —— 基本配置 ——
SAVE_PER_CLIP = True
CLIP_LEN = 81


# ============== 工具函数 ==============

def _to_pil(img: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3 and x.shape[0] in (1, 3):
            pass
        elif x.ndim == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")
        x = x.float()
        if x.min() < 0:
            x = (x + 1) / 2
        if x.max() <= 1:
            x = x * 255
        x = torch.clamp(x, 0, 255).byte()
        return to_pil_image(x)
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.min() < 0:
            arr = (arr + 1) / 2
        if arr.max() <= 1:
            arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(img)}")


def ensure_pil_list(frames: List[Union[Image.Image, torch.Tensor, np.ndarray]]) -> List[Image.Image]:
    return [_to_pil(f) for f in frames]


def save_clip_sample_videos(clip_sample, clip_dir, fps=25):
    from diffsynth import save_video
    # 视频命名映射
    key_name_map = {
        "vace_reference_image": "all_ref.mp4",
        "video": "all_gt.mp4",
        "vace_video": "all_control.mp4",
        "vace_video_mask": "all_handpose.mp4",
    }
    for key, value in clip_sample.items():
        try:
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (Image.Image, torch.Tensor, np.ndarray)):
                    frames = ensure_pil_list(value)
                    filename = key_name_map.get(key, f"{key}.mp4")
                    save_video(frames, os.path.join(clip_dir, filename), fps=fps, quality=8)
            elif isinstance(value, (Image.Image, torch.Tensor, np.ndarray)):
                img = _to_pil(value)
                img.save(os.path.join(clip_dir, f"clip_{key}.png"))
        except Exception as e:
            print(f"[WARN] Skip saving key={key}: {e}")


# ============== 核心推理逻辑 ==============

def worker(rank, gpu_id, dataset, total_gpus, output_dir, model_path, lora_path):
    """每个GPU独立进程，处理索引 rank, rank+G, rank+2G,..."""
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] starting...")

    # 初始化模型
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        offload_device="cpu"),
        ],
    )

    set_stand_in(
        pipe.vace,
        model_path=None,
        train=False
    )

    state_dict = load_state_dict(model_path)
    pipe.vace.load_state_dict(state_dict)
    pipe.load_lora(pipe.vace, lora_path, alpha=1)
    pipe.vace.requires_grad_(False)
    
    pipe.enable_vram_management()

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景"
    )

    os.makedirs(output_dir, exist_ok=True)

    for idx in range(rank, len(dataset), total_gpus):
        # if idx <= 7:
        #     continue
        sample = dataset[idx]
        print(f"[GPU {gpu_id}] Processing sample {idx+1}/{len(dataset)}")

        prompt = sample['prompt']
        total_frames = len(sample['vace_video'])
        num_clips = (total_frames + CLIP_LEN - 1) // CLIP_LEN
        agg = {'gen': []}

        for clip_id in range(num_clips):
            s = clip_id * (CLIP_LEN-1)
            e = min(s + CLIP_LEN, total_frames)
            clip_sample = {}
            for k, v in sample.items():
                if k == "vace_reference_image":
                    clip_sample[k] = v
                elif isinstance(v, list):
                    clip_sample[k] = v[s:e]
                else:
                    clip_sample[k] = v
            
            # ====== 首帧续推逻辑：除第一段以外 ======
            if clip_id > 0 and len(agg['gen']) > 0:
                # 上一段推理结果的最后一帧
                prev_last_frame = agg['gen'][-1]
                prev_last_frame = _to_pil(prev_last_frame)

                # 替换当前 clip 的 vace_video 首帧
                if 'vace_video' in clip_sample and len(clip_sample['vace_video']) > 0:
                    # 尺寸对齐（以当前 vace_video 首帧为基准）
                    tgt_w, tgt_h = clip_sample['vace_video'][0].size
                    if prev_last_frame.size != (tgt_w, tgt_h):
                        prev_last_frame_resized = prev_last_frame.resize((tgt_w, tgt_h), Image.BICUBIC)
                    else:
                        prev_last_frame_resized = prev_last_frame
                    clip_sample['vace_video'][0] = prev_last_frame_resized

                # vace_video_mask 首帧置为全黑
                if 'vace_video_mask' in clip_sample and clip_sample['vace_video_mask'] is not None:
                    if len(clip_sample['vace_video_mask']) > 0:
                        mask0 = clip_sample['vace_video_mask'][0]
                        mask0 = _to_pil(mask0)
                        # 使用原 mask 的尺寸和 mode 生成黑图
                        black_mask = Image.new(mask0.mode, mask0.size, 0)
                        clip_sample['vace_video_mask'][0] = black_mask

            # ====== 保存当前 clip 的输入（方便 debug） ======
            clip_dir = os.path.join(output_dir, f"sample_{idx:04d}_clip{clip_id:02d}")
            os.makedirs(clip_dir, exist_ok=True)
            save_clip_sample_videos(clip_sample, clip_dir, fps=25)

            w, h = clip_sample['vace_video'][0].size
            out_video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                vace_video=clip_sample['vace_video'],
                vace_reference_image=clip_sample['vace_reference_image'],
                vace_video_mask=clip_sample.get('vace_video_mask', None),
                num_frames=len(clip_sample['vace_video']),
                num_inference_steps=1,
                seed=1024,
                tiled=True,
                width=w,
                height=h,
            )

            if SAVE_PER_CLIP:
                save_video(out_video, os.path.join(clip_dir, "all_generated.mp4"), fps=25, quality=8)

            agg['gen'].extend(out_video)

        all_dir = os.path.join(output_dir, f"sample_{idx:04d}_allclips")
        os.makedirs(all_dir, exist_ok=True)
        save_video(agg['gen'], os.path.join(all_dir, "all_generated.mp4"), fps=25, quality=8)

    print(f"[GPU {gpu_id}] done.")


# ============== 主入口 ==============

def run_inference_multi(output_dir, data_csv, max_num_frames, is_fl, model_path, lora_path, devices: List[int] = None):
    """多GPU并行推理，统一保存到output_dir"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    total_gpus = len(devices)
    assert total_gpus > 0, "No available GPUs found."

    print(f"Detected {total_gpus} GPUs -> {devices}")

    dataset = HumanHoiDataset_inference(
        data_dir=data_csv,
        video_size=(1280, 720),
        fps=25,
        max_num_frames=max_num_frames,
        skip_frms_num=3,
        ref_id_type="random",
        draw_hand_color="ori",
        info_class="dwpose_test123",
        data_aug=False,
        is_random=False,
        ref_first_frame=True,
        ref_img=None,
        is_test=True,
        scale=1,
        ref_in_bbox = False,
        ref_bg = 128,
        is_fl=is_fl,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = []
    for rank, gpu_id in enumerate(devices):
        p = ctx.Process(target=worker, args=(rank, gpu_id, dataset, total_gpus, OUTPUT_DIR, model_path, lora_path))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("✅ All GPUs finished inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video inference with multi-GPU")
    parser.add_argument('--output_dir', type=str, default="results/swap",
                        help="Output directory for generated videos")
    parser.add_argument('--data_csv', type=str, default="/root/paddlejob/workspace/huangxuan/swap_data_f50/long_video_swap/swap.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Comma-separated list of GPU indices to use (e.g., '0,1,2,3')")
    parser.add_argument('--max_num_frames', type=int, default=81,
                        help="Maximum number of frames to process")
    parser.add_argument('--is_fl', action='store_true', default=False,
                        help="Enable first-last frame mode")
    parser.add_argument('--model_path', type=str, default="models/GenHOI_VACE/step-5000-gate_attn.safetensors",
                        help="Path to the model state dict weights")
    parser.add_argument('--lora_path', type=str, default="models/GenHOI_VACE/step-1700-lora-gate-720.safetensors",
                        help="Path to the LoRA model weights")
    args = parser.parse_args()

    # Parse GPU indices
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    
    run_inference_multi(args.output_dir, args.data_csv, args.max_num_frames, args.is_fl, args.model_path, args.lora_path, devices=gpu_list)

# python examples/wanvideo/swap_infer.py \
#     --output_dir results/swap_81_vace \
#     --data_csv demo/demo.csv\
#     --gpus 3 \
#     --max_num_frames 81

# python examples/wanvideo/swap_infer.py \
#     --output_dir results/swap_401 \
#     --data_csv data/long_video_swap/swap_f16.csv \
#     --gpus 0,1,2,3 \
#     --max_num_frames 401 \
#     --is_fl

# python examples/wanvideo/swap_infer.py \
#     --output_dir results/swap_81_vace_flf \
#     --data_csv data/long_video_swap/swap.csv \
#     --max_num_frames 81 \
#     --gpus 4,5,6,7 \
#     --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
#     --lora_path models/GenHOI_VACE/step-1100-lora-gate-flf-720-2.safetensors \
#     --is_fl