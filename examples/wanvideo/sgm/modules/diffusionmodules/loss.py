from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from ...modules.diffusionmodules.sampling import VideoDDIMSampler, VPSDEDPMPP2MSampler
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS

# import rearrange
from einops import rearrange
import random
from sat import mpu
from loguru import logger


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        # cond = conditioner(batch)
        cond = None
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx
        additional_model_inputs["poses"] = batch["render_frames"] #[1,39,3,H,W]
        additional_model_inputs["ref_image"] = batch["ref_image"] #[1,1,16,h,w]
        additional_model_inputs['pred_motion_frames'] = batch["pred_motion_frames"]
        if 'inpaint_x' in batch.keys():
            # print('inpaint')
            additional_model_inputs['inpaint_x'] = batch["inpaint_x"]
        if 'gt_num' in batch.keys():
            additional_model_inputs['gt_num'] = batch['gt_num']
        # import ipdb;ipdb.set_trace()
        b = batch["render_frames"].shape[0]
        cfg_mask = torch.zeros(b, device=batch["ref_image"].device)
        cfg_pose_mask = torch.bernoulli(cfg_mask, 0.9).view(b, 1, 1, 1, 1) # 1 - 0.1
        cfg_ref_mask = torch.bernoulli(cfg_mask, 0.9).view(b, 1, 1, 1) # 1 - 0.1
        additional_model_inputs["poses"] = additional_model_inputs["poses"] * cfg_pose_mask
        additional_model_inputs["ref_image"] = additional_model_inputs["ref_image"] * cfg_ref_mask

        # TODO add cfg for motion frames, 随机drop motion frames
        if 'motion_frames' in batch.keys():
            motion_mask = torch.bernoulli(cfg_mask, 0.5).view(b, 1, 1, 1, 1)
            additional_model_inputs['motion_frames'] = batch['motion_frames'] #[1,2,16,h,w]
            additional_model_inputs["motion_frames"] = additional_model_inputs["motion_frames"] * motion_mask * cfg_ref_mask
        
        try:
            additional_model_inputs['motion_len'] = batch['motion_len']
            additional_model_inputs['motion_tokens_num'] = batch['motion_tokens_num']
        except:
            # continue
            pass

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


def get_3d_position_ids(frame_len, h, w):
    i = torch.arange(frame_len).view(frame_len, 1, 1).expand(frame_len, h, w)
    j = torch.arange(h).view(1, h, 1).expand(frame_len, h, w)
    k = torch.arange(w).view(1, 1, w).expand(frame_len, h, w)
    position_ids = torch.stack([i, j, k], dim=-1).reshape(-1, 3)
    return position_ids
