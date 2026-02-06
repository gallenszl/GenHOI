"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from .sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from ...util import SeededNoise

from .guiders import DynamicCFG

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]]).float()

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class VideoDDIMSampler(BaseDiffusionSampler):
    def __init__(self, fixed_frames=0, sdedit=False, **kwargs):
        super().__init__(**kwargs)
        self.fixed_frames = fixed_frames
        self.sdedit = sdedit

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
            return_idx=True,
            do_append_zero=False,
        )
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1]) - 1, torch.tensor(list(timesteps))])

        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def denoise(self, x, input_data, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None):
        additional_model_inputs = {}

        # print('x:', x.shape)
        # for key, value in input_data.items():
        #     if torch.is_tensor(value):
        #         additional_model_inputs[key] = torch.cat([value, value])
        #         print(key, ':', additional_model_inputs[key].shape)
        # if isinstance(scale, torch.Tensor) == False and scale == 1:
        #     additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep
        #     if scale_emb is not None:
        #         additional_model_inputs["scale_emb"] = scale_emb
        #     denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        # else:
        #     additional_model_inputs["idx"] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
        #     denoised = denoiser(
        #         *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
        #     ).to(torch.float32)
        #     if isinstance(self.guider, DynamicCFG):
        #         denoised = self.guider(
        #             denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, step_index=self.num_steps - timestep, scale=scale
        #         )
        #     else:
        #         denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, scale=scale)
        for key, value in input_data.items():
            if torch.is_tensor(value):
                additional_model_inputs[key] = value
        additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep
        if scale_emb is not None:
            additional_model_inputs["scale_emb"] = scale_emb
        denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        return denoised

    def sampler_step(
        self,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        input_data,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        denoised = self.denoise(
            x, input_data, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)

        a_t = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        return x

    def __call__(self, denoiser, x, input_data, cond, uc=None, num_steps=None, scale=None, scale_emb=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        # for i in self.get_sigma_gen(num_sigmas):
        i = kwargs["step"]
        x = self.sampler_step(
            s_in * alpha_cumprod_sqrt[i],
            s_in * alpha_cumprod_sqrt[i + 1],
            denoiser,
            x,
            input_data,
            cond,
            uc,
            idx=self.num_steps - i,
            timestep=timesteps[-(i + 1)],
            scale=scale,
            scale_emb=scale_emb,
        )

        return x  

class VPODEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        mult2 = (-h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        input_data,
        cond,
        uc=None,
        idx=None,
        timestep=None,
    ):
        denoised = self.denoise(x, input_data, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, input_data, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        i = kwargs["step"]

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                input_data,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
            )

        return x
    