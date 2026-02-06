#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Standalone FVD (Fréchet Video Distance) computation module.
Extracted from DisCo project for easier integration.

Usage:
    python tools/eval_fvd/fvd.py \
        --gen_dir /path/to/generated_videos \
        --gt_dir /path/to/ground_truth_videos \
        --sample_duration 81 \
        --output metrics_fvd.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from PIL import Image, ImageSequence
import ffmpeg

# Local imports
from resize import make_resizer
from resnet3d import resnet50
from inception3d import InceptionI3d


# =============================================================================
# Core FVD Computation Functions
# =============================================================================

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def fid_from_feats(feats1, feats2):
    """Compute FID score from feature arrays."""
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


# =============================================================================
# Feature Extractor Builder
# =============================================================================

def build_feature_extractor(mode, weights_dir, device=torch.device("cuda"), sample_duration=16):
    """Build a feature extractor model for FVD computation."""
    if mode == "FVD-3DRN50":
        feat_model = resnet50(
            num_classes=400, shortcut_type="B",
            sample_size=112, sample_duration=sample_duration, last_fc=False
        )
        model_path = os.path.join(weights_dir, "resnet-50-kinetics.pth")
        model_data = torch.load(model_path, map_location='cpu')
        model_state_new = {}
        for key, value in model_data['state_dict'].items():
            key_new = key.replace('module.', '')
            model_state_new[key_new] = value
        feat_model.load_state_dict(model_state_new)
        feat_model = feat_model.to(device).eval()

    elif mode == "FVD-3DInception":
        feat_model = InceptionI3d(400, in_channels=3)
        model_path = os.path.join(weights_dir, "i3d_pretrained_400.pt")
        feat_model.load_state_dict(torch.load(model_path))
        feat_model = feat_model.to(device).eval()
    else:
        raise NotImplementedError(f"Mode {mode} not supported")

    return feat_model


# =============================================================================
# Video Dataset Classes
# =============================================================================

def gif_to_nparray(gif_path):
    """Convert GIF to numpy array."""
    gif = Image.open(gif_path)
    frames = [np.array(frame.copy().convert('RGB'), dtype=np.uint8) for frame in ImageSequence.Iterator(gif)]
    video = np.stack(frames)
    return video


class DatasetFVDVideoResize(torch.utils.data.Dataset):
    """Dataset for loading and preprocessing videos for FVD computation."""

    def __init__(self, files, sample_duration=16, mode='FVD-3DRN50', img_size=112, return_name=False):
        self.files = files
        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
        self.resize_func = make_resizer("PIL", False, "bicubic", (img_size, img_size))
        self.return_name = return_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            path = str(self.files[i])

            if Path(path).suffix == '.gif':
                video = gif_to_nparray(path)
            else:
                probe = ffmpeg.probe(path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                out, _ = (ffmpeg.input(path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True))
                video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

            # Resize video frames
            video_resize = []
            for vim in video:
                vim_resize = self.resize_func(vim)
                video_resize.append(vim_resize)

            video = np.stack(video_resize, axis=0)
            video = torch.as_tensor(video.copy()).float()
            num_v = video.shape[0]

            if num_v % self.sample_duration != 0 and self.mode == "FVD-3DRN50":
                num_v_ag = self.sample_duration - num_v % self.sample_duration
                video_aug = video[[-1], :, :, :].repeat(num_v_ag, 1, 1, 1)
                video = torch.cat([video, video_aug], dim=0)
                num_seg = num_v // self.sample_duration + 1
            else:
                num_seg = num_v // self.sample_duration

            if self.mode == 'FVD-3DRN50':
                video = video - self.pixel_mean
                video = video.view(num_seg, self.sample_duration, self.img_size, self.img_size, 3).contiguous().permute(0, 4, 1, 2, 3).float()
            elif self.mode == "FVD-3DInception":
                video = video / 127.5 - 1
                video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()

            if self.return_name:
                return video, Path(path).stem
            return video
        except Exception as e:
            print(f'{i} skipped because {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def get_batch_features(batch, model, device):
    """Get features for a batch of inputs."""
    if model is None:
        return batch.numpy()
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


def compute_3d_video_prediction(dataset, feat_model, batch_size, num_workers, device):
    """Compute 3D video features using the given model."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    assert batch_size == 1

    l_feats = []
    for batch in tqdm(dataloader, desc='Extracting video features'):
        batch = batch.squeeze(0)
        l_feats.append(get_batch_features(batch, feat_model, device).mean(0, keepdims=True))

    np_feats = np.concatenate(l_feats)
    return np_feats


# =============================================================================
# Main FVD Computation
# =============================================================================

def compute_fvd(gen_files, gt_files, mode, sample_duration, weights_dir, device, batch_size=1, num_workers=8):
    """
    Compute FVD score between generated and ground truth videos.
    
    Args:
        gen_files: List of paths to generated video files
        gt_files: List of paths to ground truth video files
        mode: Feature extractor mode ('FVD-3DRN50' or 'FVD-3DInception')
        sample_duration: Number of frames to sample from each video
        weights_dir: Directory containing pretrained model weights
        device: Computation device
        batch_size: Batch size for feature extraction
        num_workers: Number of data loader workers
    
    Returns:
        FVD score (float)
    """
    # Determine sample size based on mode
    if mode == "FVD-3DRN50":
        sample_size = 112
    elif mode == "FVD-3DInception":
        sample_size = 224
    else:
        raise NotImplementedError(f"Mode {mode} not supported")

    # Build feature extractor
    feat_model = build_feature_extractor(mode, weights_dir, device, sample_duration)

    # Build datasets
    print(f"Using raw video gen dataset for FVD computation, first file suffix: {Path(gen_files[0]).suffix}")
    dataset_gen = DatasetFVDVideoResize(gen_files, sample_duration, mode, sample_size)

    print(f"Using raw video gt dataset for FVD computation, first file suffix: {Path(gt_files[0]).suffix}")
    dataset_gt = DatasetFVDVideoResize(gt_files, sample_duration, mode, sample_size)

    # Extract features
    np_feats_gen = compute_3d_video_prediction(dataset_gen, feat_model, batch_size, num_workers, device)
    np_feats_gt = compute_3d_video_prediction(dataset_gt, feat_model, batch_size, num_workers, device)

    # Compute FVD
    fvd_score = fid_from_feats(feats1=np_feats_gen, feats2=np_feats_gt)

    return fvd_score


def get_video_files(directory, extensions=('.mp4', '.gif', '.avi', '.mov')):
    """Get all video files from a directory."""
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, f'*{ext}')))
    return sorted(files)


def compute_fvd_from_dirs(gen_dir, gt_dir, sample_duration, weights_dir, device='cuda',
                          batch_size=1, num_workers=8, modes=None):
    """
    Compute FVD scores from directories of videos.
    
    Args:
        gen_dir: Directory containing generated videos
        gt_dir: Directory containing ground truth videos
        sample_duration: Number of frames per video
        weights_dir: Directory containing model weights
        device: Computation device
        batch_size: Batch size
        num_workers: Number of workers
        modes: List of modes to compute ('FVD-3DRN50', 'FVD-3DInception')
    
    Returns:
        Dictionary of FVD scores
    """
    if modes is None:
        modes = ['FVD-3DRN50', 'FVD-3DInception']

    # Get video files
    gen_files = get_video_files(gen_dir)
    gt_files = get_video_files(gt_dir)

    print(f"Found {len(gen_files)} generated videos and {len(gt_files)} GT videos")

    if len(gen_files) == 0 or len(gt_files) == 0:
        raise ValueError("No video files found in one or both directories")

    if len(gen_files) != len(gt_files):
        print(f"Warning: Number of generated ({len(gen_files)}) and GT ({len(gt_files)}) videos differ")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    results = {}

    for mode in modes:
        print(f"\nComputing {mode}...")
        print(f"start evluation {mode} over {len(gen_files)} generated Image/Video and {len(gt_files)} gt Image/Video")
        
        fvd_score = compute_fvd(
            gen_files, gt_files, mode, sample_duration, weights_dir, device, batch_size, num_workers
        )
        
        print(f"The {mode} {fvd_score}, duration:{sample_duration}")
        results[mode] = fvd_score

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute FVD (Fréchet Video Distance)')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='Directory containing generated videos')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth videos')
    parser.add_argument('--sample_duration', type=int, default=16,
                        help='Number of frames to sample from each video')
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='Directory containing model weights (default: same as this script)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--modes', type=str, nargs='+', default=['FVD-3DRN50', 'FVD-3DInception'],
                        choices=['FVD-3DRN50', 'FVD-3DInception'],
                        help='FVD modes to compute')

    args = parser.parse_args()

    # Set default weights directory
    if args.weights_dir is None:
        args.weights_dir = os.path.dirname(os.path.abspath(__file__))

    # Compute FVD
    results = compute_fvd_from_dirs(
        gen_dir=args.gen_dir,
        gt_dir=args.gt_dir,
        sample_duration=args.sample_duration,
        weights_dir=args.weights_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modes=args.modes
    )

    print("\n" + "="*50)
    print("Results:")
    print(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
