import os
import argparse
import random
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from diffsynth import ModelManager, WanVideoPipelineRope, save_video, save_frames
from dataset.customer_dataset import HumanHoiDataset_inference
from dataset.customer_dataset_anchorcrafter import HumanHoiDataset_anchorcrafter
import torchvision.transforms as transforms
from PIL import Image

from diffsynth.models.set_condition_branch import set_attn_gate


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29979'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def stitch_and_save(agg, all_dir, fps=25, quality=8):
    """
    将 agg 中的 gt, ctrl, gen, repl 列表按 2x2 网格拼接，并保存为视频。
    agg: dict 包含四个 list: 'gt', 'ctrl', 'gen', 'repl'
    all_dir: 输出目录
    """
    # 拼接帧
    stitched_frames = []
    for im_gt, im_ctrl, im_gen, im_repl in zip(agg['gt'], agg['ctrl'], agg['gen'], agg['repl']):
        im0 = im_gt.convert('RGB')
        im1 = im_ctrl.convert('RGB')
        im2 = im_gen.convert('RGB')
        im3 = im_repl.convert('RGB')

        w, h = im0.size
        grid = Image.new('RGB', (w * 2, h * 2))
        grid.paste(im0, (0,     0))
        grid.paste(im1, (w,     0))
        grid.paste(im2, (0,     h))
        grid.paste(im3, (w,     h))
        stitched_frames.append(grid)

    # 保存视频
    save_video(
        stitched_frames,
        os.path.join(all_dir, "all_stitched_2x2.mp4"),
        fps=fps,
        quality=quality
    )


def run_inference(rank, world_size, model_path, output_dir, data_csv, max_num_frames, is_fl):
    setup(rank, world_size)

    # Initialize model manager and load models
    device = torch.device(f'cuda:{rank}')
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
    model_manager.load_models(["models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors".split(",")])
    model_manager.load_models([
        'models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
    ], torch_dtype=torch.float32)
    model_manager.load_models([
        'models/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth',
        'models/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth'
    ])

    # Build pipeline
    pipe = WanVideoPipelineRope.from_model_manager(model_manager, device=device)

    set_attn_gate(
        pipe,
        model_path=model_path,
        train=False
    )

    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Config
    negative_prompt = "Bright tones, overexposed, static, blurred details, a rotating hand, floating..."
    random_selection = False
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
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
        is_fl=is_fl
    )

    n_samples = len(dataset)
    all_indices = random.sample(range(n_samples), n_samples) if random_selection else list(range(n_samples))
    selected_indices = [idx for i, idx in enumerate(all_indices) if i % world_size == rank]

    for global_pos, idx in enumerate(selected_indices):
        # if idx<=3:
        #     continue
        # try:
        print(f"[Rank {rank}] Sample {global_pos+1}/{len(selected_indices)} (idx={idx})")
        sample = dataset[idx]
        prompt = sample['prompt']
    # prompt = "In the video, the host’s hand moves without any rotation while showcasing the product."
        num_frames = sample['gt_frames'].shape[0]
        num_clips = (num_frames-1) // 80
        if num_frames<80:
            num_clips = 1
        video_last_frame = None

        agg = {'gen': [], 'ctrl': [], 'repl': [], 'gt': [], 'ref': [], 'hand': []}

        for clip_id in range(num_clips):
            # 切片数据
            clip_sample = {}
            for k, v in sample.items():
                if k in ["num_frames", "prompt", "fps", "data_path", "pixel_values_ref_img"]:
                    clip_sample[k] = v
                else:
                    if clip_id == 0:
                        clip_sample[k] = v[:81]
                    else:
                        clip_sample[k] = v[clip_id*80:(clip_id+1)*80+1]
            clip_sample['num_frames'] = clip_sample.get('num_frames', clip_sample['gt_frames'].shape[0])

            if video_last_frame is not None:
                clip_sample['wo_obj_video_frames'][0] = video_last_frame
                clip_sample['pixel_values_ref_img_only_mask'][0] = -1

            # 推理
            video, video_control, video_gt, video_ref, latents_hand_pose, video_replaced = pipe(
                prompt=prompt,
                tracking_maps=clip_sample,
                negative_prompt=negative_prompt,
                input_image=None,
                image_emb_external=None,
                num_inference_steps=50,
                # seed=random.randint(0, 999999),
                seed=1024,
                num_frames=81,
                # num_frames=17,
                tiled=True,
                height=1280,
                width=720,
                # height=320,
                # width=320,
            )
            video_last_frame = transforms.functional.pil_to_tensor(video[-1]) / 255.0 * 2 - 1.0

            # 聚合
            if clip_id == 0:
                agg['gen'].extend(video)
                agg['ctrl'].extend(video_control)
                agg['repl'].extend(video_replaced)
                agg['gt'].extend(video_gt)
                agg['ref'].extend(video_ref)
                agg['hand'].extend(latents_hand_pose)
            else:
                agg['gen'].extend(video[1:])
                agg['ctrl'].extend(video_control[1:])
                agg['repl'].extend(video_replaced[1:])
                agg['gt'].extend(video_gt[1:])
                agg['ref'].extend(video_ref[1:])
                agg['hand'].extend(latents_hand_pose[1:])


        # 所有 clip 聚合后整体保存
        all_dir = os.path.join(output_dir, f"sample_{idx:04d}_allclips")
        os.makedirs(all_dir, exist_ok=True)

        # 分别保存各路视频和帧
        save_video(agg['gen'],  os.path.join(all_dir, "all_generated.mp4"), fps=25, quality=8)
        save_video(agg['ctrl'], os.path.join(all_dir, "all_control.mp4"),   fps=25, quality=8)
        save_video(agg['repl'], os.path.join(all_dir, "all_replaced.mp4"),  fps=25, quality=8)
        save_video(agg['gt'],   os.path.join(all_dir, "all_gt.mp4"),        fps=25, quality=8)
        save_video(agg['ref'],  os.path.join(all_dir, "all_ref.mp4"),       fps=25, quality=8)
        save_video(agg['hand'], os.path.join(all_dir, "all_handpose.mp4"),  fps=25, quality=8)

        # save_frames(agg['gen'],  os.path.join(all_dir, "images_generated"))
        # save_frames(agg['ctrl'], os.path.join(all_dir, "images_control"))
        # save_frames(agg['repl'], os.path.join(all_dir, "images_replaced"))
        # save_frames(agg['gt'],   os.path.join(all_dir, "images_gt"))
        # save_frames(agg['ref'],  os.path.join(all_dir, "images_ref"))

        # 拼接 2x2 并保存
        stitch_and_save(agg, all_dir, fps=25, quality=8)
        # except Exception as e:
        #     print(f"Error processing sample {idx}: {e}")

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run video inference with distributed training")
    parser.add_argument('--model_path', type=str, default="models/ckpt/first_frame_rope.consolidated",
                        help="Path to the model checkpoint")
    parser.add_argument('--output_dir', type=str, default="1015/i2v_anchorcrafter_81f_rope_mochu_sft_swap_ff_sft",
                        help="Output directory for generated videos")
    parser.add_argument('--data_csv', type=str, default="/root/paddlejob/workspace/huangxuan/swap_data_f50/long_video_swap_wofirstframe/swap.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument('--gpus', type=str, default="0,1",
                        help="Comma-separated list of GPU indices to use (e.g., '0,1,2,3')")
    parser.add_argument('--max_num_frames', type=int, default=81,
                        help="Maximum number of frames to process")
    parser.add_argument('--is_fl', action='store_true', default=False,
                        help="Enable first-last frame mode")
    args = parser.parse_args()

    # Parse GPU indices and set CUDA_VISIBLE_DEVICES
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(gpu_list)

    spawn(run_inference, args=(world_size, args.model_path, args.output_dir, args.data_csv, args.max_num_frames, args.is_fl), nprocs=world_size, join=True)



# python examples/wanvideo/test_swap.py \
#     --model_path models/GenHOI_wan_flf.consolidated \
#     --output_dir results/demo \
#     --data_csv demo/demo.csv \
#     --gpus 0 \
#     --max_num_frames 401 \
#     --is_fl

# python examples/wanvideo/test_swap.py \
#     --model_path models/GenHOI_wan_flf.consolidated \
#     --output_dir results/swap_81 \
#     --data_csv data/long_video_swap/swap.csv \
#     --gpus 2 \
#     --max_num_frames 81 \
#     --is_fl

# python examples/wanvideo/test_swap.py \
#     --model_path models/GenHOI_wan_flf.consolidated \
#     --output_dir results/swap_401 \
#     --data_csv data/long_video_swap/swap_f16.csv \
#     --gpus 3 \
#     --max_num_frames 401 \
#     --is_fl