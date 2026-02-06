from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from torchvision import transforms

import torch
import numpy as np
import cv2
from pathlib import Path

def visualize_latent_feature_grid(
    latent: torch.Tensor,
    output_path: str = "latent_feature_grid.mp4",
    fps: int = 10,
    grid_rows: int = 4,
    grid_cols: int = 4
):
    """
    将 latent feature 可视化为灰度视频，每个通道独立显示，拼接为网格输出视频。
    
    参数:
        latent (torch.Tensor): 形状为 [1, C, T, H, W] 的 latent 特征图。
        output_path (str): 输出视频路径。
        fps (int): 视频帧率。
        grid_rows (int): 可视化网格行数（通常为 sqrt(C)）。
        grid_cols (int): 可视化网格列数。
    """
    assert latent.dim() == 5 and latent.shape[0] == 1, "输入 latent 必须是 [1, C, T, H, W]"
    _, C, T, H, W = latent.shape
    latent = latent[0]  # 去掉 batch 维度，变成 [C, T, H, W]

    # Normalize to [0, 255]
    def normalize_to_uint8(tensor):
        tensor = tensor.clone()
        tensor -= tensor.min()
        tensor /= (tensor.max() + 1e-6)
        return (tensor * 255).byte()

    # 初始化视频写入器
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_h, frame_w = H, W
    video_h, video_w = grid_rows * frame_h, grid_cols * frame_w
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h), isColor=False)

    for t in range(T):
        frame_list = []
        for c in range(C):
            feature_map = latent[c, t]  # [H, W]
            img = normalize_to_uint8(feature_map).cpu().numpy()  # uint8
            frame_list.append(img)

        # 拼接为 grid
        grid_frame = []
        for i in range(grid_rows):
            row = np.hstack(frame_list[i * grid_cols:(i + 1) * grid_cols])
            grid_frame.append(row)
        grid = np.vstack(grid_frame)

        writer.write(grid)

    writer.release()
    print(f"可视化保存至: {output_path}")


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.to_pil = transforms.ToPILImage()

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])   ### clip to extract feature
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        # import pdb; pdb.set_trace()
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def resize_mask(self, msk, height, width):
        # import pdb; pdb.set_trace()
        B, F, H, W = msk.shape
        msk = msk.view(B * F, 1, H, W)
        msk = torch.nn.functional.interpolate(msk, size=(H // 8, W // 8), mode='nearest')
        msk = msk.view(B, F, H // 8, W // 8)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(-1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)
        return msk


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        image_emb_external=None,
        input_image=None,
        input_video=None,
        tracking_maps=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        # if image_emb_external is None:
        #     if input_image is not None and self.image_encoder is not None:
        #         self.load_models_to_device(["image_encoder", "vae"])
        #         image_emb = self.encode_image(input_image, num_frames, height, width)
        #     else:
        #         image_emb = {}
        # else:
        #     image_emb = image_emb_external
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        if tracking_maps is not None:
            self.load_models_to_device(['vae'])
            gt_frames = tracking_maps["gt_frames"].to(dtype=noise.dtype, device=self.device).unsqueeze(0).permute(0, 2, 1, 3, 4)
            wo_obj_video_frames = tracking_maps["wo_obj_video_frames"].to(dtype=noise.dtype, device=self.device).unsqueeze(0).permute(0, 2, 1, 3, 4)
            pixel_values_ref_img = tracking_maps["pixel_values_ref_img"].unsqueeze(0).to(dtype=noise.dtype, device=self.device).permute(0, 2, 1, 3, 4)
            pixel_values_ref_img_only_mask = tracking_maps["pixel_values_ref_img_only_mask"].unsqueeze(0).to(dtype=noise.dtype, device=self.device).permute(0, 2, 1, 3, 4)
            # pixel_values_ref_img = pixel_values_ref_img.repeat(1, gt_frames.shape[2], 1, 1, 1).permute(0, 2, 1, 3, 4)
            # hand_pose = tracking_maps["hand_pose"].unsqueeze(0).to(dtype=noise.dtype, device=self.device).permute(0, 2, 1, 3, 4)
            # hand_obj_box = tracking_maps["hand_obj_box"].unsqueeze(0).to(dtype=noise.dtype, device=self.device).permute(0, 2, 1, 3, 4)
            dense_mask = pixel_values_ref_img_only_mask.clone()
            # import pdb; pdb.set_trace()
            dense_mask = dense_mask[:, 0, :, :, :]
            dense_mask[dense_mask == -1] = 0 
            _, _, num_frames, height, width = gt_frames.shape
            mask_input_for_v2v = self.resize_mask(dense_mask, height, width)

            clip_feature = None
            y = None
            for i in range(gt_frames.shape[0]):
                pil_image = self.to_pil((wo_obj_video_frames[i, :, 0].float() + 1) / 2)
                # image_zero = Image.new(mode=pil_image.mode, size=pil_image.size)
                # print(np.array(image_zero))
                # import pdb; pdb.set_trace()
                image_feature = self.encode_image(pil_image, num_frames, height, width)
                _clip_feature = image_feature["clip_feature"].to(self.device)
                _y = image_feature["y"].to(self.device)
                if clip_feature is None:
                    clip_feature = _clip_feature
                    # y = _y
                else:
                    clip_feature = torch.cat([clip_feature, _clip_feature], dim=0)
                    # y = torch.cat([y, _y], dim=0)
                    # print(clip_feature.shape, y.shape) 

            # import pdb; pdb.set_trace()
            # latents_hand_pose = self.encode_video(hand_pose, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            # import pdb; pdb.set_trace()
            latents_wo_obj_video_frames = self.encode_video(wo_obj_video_frames, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            # hand_obj_box = self.encode_video(hand_obj_box, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            # import pdb; pdb.set_trace()
            pixel_values_ref_img = self.encode_video(pixel_values_ref_img, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            latents_gt_frames = self.encode_video(gt_frames, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            pixel_values_ref_img_only_mask = self.encode_video(pixel_values_ref_img_only_mask, **tiler_kwargs)[0].unsqueeze(0).to(dtype=noise.dtype, device=noise.device)
            control_information = {
                "latents_wo_obj_video_frames": latents_wo_obj_video_frames,
                "pixel_values_ref_img": pixel_values_ref_img,
                "pixel_values_ref_img_mask": pixel_values_ref_img_only_mask,
            }
            y = torch.cat([mask_input_for_v2v, latents_wo_obj_video_frames], dim=1)
        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(latents, timestep=timestep, clip_feature=clip_feature, y=y, **prompt_emb_posi, **extra_input)
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(latents, timestep=timestep, clip_feature=clip_feature, y=y, **prompt_emb_nega, **extra_input)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])
        # visualize_latent_feature_grid(latents, output_path="vis/latent_vis.mp4", fps=10, grid_rows=4, grid_cols=4)
        # visualize_latent_feature_grid(latents_wo_obj_video_frames, output_path="vis/latents_wo_obj_video_frames.mp4", fps=10, grid_rows=4, grid_cols=4)
        # visualize_latent_feature_grid(latents_gt_frames, output_path="vis/latents_gt_frames.mp4", fps=10, grid_rows=4, grid_cols=4)
        # visualize_latent_feature_grid(pixel_values_ref_img, output_path="vis/pixel_values_ref_img.mp4", fps=10, grid_rows=4, grid_cols=4)
        # visualize_latent_feature_grid(pixel_values_ref_img_only_mask, output_path="vis/pixel_values_ref_img_only_mask.mp4", fps=10, grid_rows=4, grid_cols=4)
        # visualize_latent_feature_grid(mask_input_for_v2v, output_path="vis/mask_input_for_v2v.mp4", fps=10, grid_rows=4, grid_cols=1)
        mask_input_for_v2v_expanded = torch.cat(
            [mask_input_for_v2v[:, i:i+1, ...].repeat(1, 4, 1, 1, 1) for i in range(4)],
            dim=1
        )
        # visualize_latent_feature_grid(mask_input_for_v2v_expanded, output_path="vis/mask_input_for_v2v_expanded.mp4", fps=10, grid_rows=4, grid_cols=4)
        latents_replaced = latents.clone()
        latents_replaced[mask_input_for_v2v_expanded<0.8] = latents_gt_frames[mask_input_for_v2v_expanded<0.8]
        # visualize_latent_feature_grid(latents_replaced, output_path="vis/latents_replaced.mp4", fps=10, grid_rows=4, grid_cols=4)

        frames = self.decode_video(latents, **tiler_kwargs) #[1, 16, 21, 104, 90]
        frames_replaced = self.decode_video(latents_replaced, **tiler_kwargs) #[1, 16, 21, 104, 90]
        frames_control = self.decode_video(latents_wo_obj_video_frames, **tiler_kwargs)
        gt_frames = self.decode_video(latents_gt_frames, **tiler_kwargs)
        pixel_values_ref_img = self.decode_video(pixel_values_ref_img, **tiler_kwargs)
        pixel_values_ref_img_only_mask = self.decode_video(pixel_values_ref_img_only_mask, **tiler_kwargs)
        # import pdb; pdb.set_trace()
        self.load_models_to_device([])
        frames_tensor = frames[0]
        frames_video = self.tensor2video(frames_tensor)
        frames_control = self.tensor2video(frames_control[0])
        frames_replaced = self.tensor2video(frames_replaced[0])
        gt_frames = self.tensor2video(gt_frames[0])
        pixel_values_ref_img = self.tensor2video(pixel_values_ref_img[0])
        pixel_values_ref_img_only_mask = self.tensor2video(pixel_values_ref_img_only_mask[0])
        # dense_mask = self.tensor2video(dense_mask[0])

        return frames_video, frames_control, gt_frames, pixel_values_ref_img, pixel_values_ref_img_only_mask, frames_replaced