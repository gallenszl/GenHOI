"""
Video metrics calculator for FVD.
Extracted and simplified from DisCo's metric_center.py

Main function: calculate_video_metrics()
"""

import os
import sys
import glob
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy import linalg

# Add current directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from feature_extractor import build_feature_extractor
from video_dataset import DatasetFVDVideoResize, DatasetFVDVideoFromFramesResize


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet distance between two multivariate Gaussians.
    
    The Frechet distance between X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability
    
    Returns:
        Frechet distance (float)
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


def get_batch_features(batch, model, device):
    """
    Extract features from a batch of video segments.
    
    Args:
        batch: Tensor of video data
        model: Feature extraction model
        device: torch device
    
    Returns:
        Numpy array of features
    """
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


def fid_from_feats(feats1, feats2):
    """
    Compute FID score from two sets of features.
    
    Args:
        feats1: Features from first set (generated)
        feats2: Features from second set (ground truth)
    
    Returns:
        FID score (float)
    """
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


def compute_3d_video_prediction(dataset, feat_model, batch_size, num_workers, device):
    """
    Extract 3D features from video dataset using a 3D model.
    
    For each video, the model processes it in segments and averages the features.
    
    Args:
        dataset: Video dataset
        feat_model: 3D feature extraction model
        batch_size: Batch size (should be 1 for video-level averaging)
        num_workers: Number of dataloader workers
        device: torch device
    
    Returns:
        Numpy array of features [num_videos, feature_dim]
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False
    )
    
    assert batch_size == 1, "Batch size must be 1 for video-level feature averaging"
    
    l_feats = []
    for batch in tqdm(dataloader, desc='Extracting 3D video features'):
        batch = batch.squeeze(0)
        # Average features across all segments of one video
        feat = get_batch_features(batch, feat_model, device).mean(0, keepdims=True)
        l_feats.append(feat)

    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_fid_video_scores(gen_inst_names_full, gt_inst_names_full, 
                             feat_model, mode, sample_duration, 
                             batch_size, num_workers, device):
    """
    Compute FID-VID or FVD score between generated and ground truth videos.
    
    Args:
        gen_inst_names_full: List of generated video paths
        gt_inst_names_full: List of ground truth video paths
        feat_model: Feature extraction model
        mode: 'FVD-3DRN50' or 'FVD-3DInception'
        sample_duration: Number of frames per segment
        batch_size: Batch size for processing
        num_workers: Number of dataloader workers
        device: torch device
    
    Returns:
        FID/FVD score (float)
    """
    if mode == "FVD-3DRN50":
        sample_size = 112
    elif mode == "FVD-3DInception":
        sample_size = 224
    else:
        raise NotImplementedError(f"Mode {mode} not supported")
    
    # Determine if inputs are videos or frame sequences
    if Path(gen_inst_names_full[0]).suffix in [".mp4", ".gif"]:
        print(f"Using raw video dataset for {mode}, first file: {Path(gen_inst_names_full[0]).suffix}")
        dataset_gen = DatasetFVDVideoResize(
            gen_inst_names_full, sample_duration, mode, sample_size
        )
    else:
        print(f"Using frame sequence dataset for {mode}, first file: {Path(gen_inst_names_full[0]).suffix}")
        dataset_gen = DatasetFVDVideoFromFramesResize(
            gen_inst_names_full, sample_duration, mode, sample_size
        )

    np_feats_gen = compute_3d_video_prediction(
        dataset_gen, feat_model, batch_size=batch_size, 
        num_workers=num_workers, device=device
    )

    if Path(gt_inst_names_full[0]).suffix in [".mp4", ".gif"]:
        print(f"Using raw video dataset for GT, first file: {Path(gt_inst_names_full[0]).suffix}")
        dataset_gt = DatasetFVDVideoResize(
            gt_inst_names_full, sample_duration, mode, sample_size
        )
    else:
        print(f"Using frame sequence dataset for GT, first file: {Path(gt_inst_names_full[0]).suffix}")
        dataset_gt = DatasetFVDVideoFromFramesResize(
            gt_inst_names_full, sample_duration, mode, sample_size
        )

    np_feats_gt = compute_3d_video_prediction(
        dataset_gt, feat_model, batch_size=batch_size,
        num_workers=num_workers, device=device
    )
    
    fid_score = fid_from_feats(feats1=np_feats_gen, feats2=np_feats_gt)
    return fid_score


def calculate_video_metrics(root_dir, path_gen, path_gt, metrics,
                            batch_size=1, sample_duration=16,
                            num_gen=None, num_gt=None, num_workers=16,
                            device='cuda'):
    """
    Calculate FID-VID and/or FVD metrics for video generation evaluation.
    
    Args:
        root_dir: Root directory (for relative paths)
        path_gen: Path to generated videos
        path_gt: Path to ground truth videos
        metrics: List of metrics to compute, e.g., ['fid-vid', 'fvd']
        batch_size: Batch size (should be 1 for video metrics)
        sample_duration: Number of frames per video segment
        num_gen: Max number of generated videos to evaluate (None = all)
        num_gt: Max number of GT videos to evaluate (None = all)
        num_workers: Number of dataloader workers
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of metric results, e.g., {'FVD-3DRN50': 123.45, 'FVD-3DInception': 234.56}
    """
    # Ensure full paths
    if root_dir not in path_gen:
        path_gen = os.path.join(root_dir, path_gen)
    if root_dir not in path_gt:
        path_gt = os.path.join(root_dir, path_gt)
    
    # Mapping of metric names
    type2metric = {
        'fvd': "FVD-3DInception"
    }
    
    device = torch.device(device)
    res_all = {}
    
    # Load video file lists
    gen_inst_names_v = []
    gt_inst_names_v = []
    
    for metric_type in metrics:
        if metric_type.lower() not in ['fvd']:
            print(f"Metric '{metric_type}' is not supported in this module. Skipping.")
            continue
        
        # Load file lists if not already loaded
        if len(gen_inst_names_v) == 0:
            gen_inst_names_v = (glob.glob(f"{path_gen}/*.gif") + 
                               glob.glob(f"{path_gen}/*.mp4") + 
                               glob.glob(f"{path_gen}/*.png") + 
                               glob.glob(f"{path_gen}/*.jpg"))
            gt_inst_names_v = (glob.glob(f"{path_gt}/*.gif") + 
                              glob.glob(f"{path_gt}/*.mp4") + 
                              glob.glob(f"{path_gt}/*.png") + 
                              glob.glob(f"{path_gt}/*.jpg"))
            
            gen_inst_names_v = sorted([os.path.basename(f) for f in gen_inst_names_v])
            gt_inst_names_v = sorted([os.path.basename(f) for f in gt_inst_names_v])
            
            if num_gen is not None:
                gen_inst_names_v = gen_inst_names_v[:num_gen]
            if num_gt is not None:
                gt_inst_names_v = gt_inst_names_v[:num_gt]
            
            if len(gen_inst_names_v) == 0 or len(gt_inst_names_v) == 0:
                print(f"Empty gen/gt folder: {path_gen}, {path_gt}")
                return res_all
            
            # Build full paths
            gen_inst_names_full_v = [os.path.join(path_gen, name) 
                                    for name in tqdm(gen_inst_names_v, desc='Loading gen paths')]
            gt_inst_names_full_v = [os.path.join(path_gt, name) 
                                   for name in tqdm(gt_inst_names_v, desc='Loading gt paths')]
        
        # Compute the metric
        mode = type2metric[metric_type.lower()]
        print(f"\n{'='*70}")
        print(f"Computing {mode} ({metric_type})")
        print(f"{'='*70}")
        
        # Build feature extractor
        feat_model = build_feature_extractor(mode=mode, device=device, 
                                            sample_duration=sample_duration)
        
        # Compute score
        v_batch_size = min(batch_size, min(len(gen_inst_names_v), len(gt_inst_names_v)))
        score = compute_fid_video_scores(
            gen_inst_names_full_v, gt_inst_names_full_v, feat_model,
            mode=mode, sample_duration=sample_duration,
            batch_size=v_batch_size, num_workers=num_workers, device=device
        )
        
        res_all[mode] = float(score)
        print(f"\n{mode} score: {score:.4f}")
    
    return res_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate FID-VID and FVD metrics for video generation"
    )
    parser.add_argument('--root_dir', type=str, required=True,
                       help="Root directory for relative paths")
    parser.add_argument('--path_gen', type=str, required=True,
                       help="Path to generated videos (relative to root_dir)")
    parser.add_argument('--path_gt', type=str, required=True,
                       help="Path to ground truth videos (relative to root_dir)")
    parser.add_argument('--type', type=str, default='fvd', nargs="+",
                       help="Metrics to compute: fvd")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size (should be 1 for video metrics)")
    parser.add_argument('--sample_duration', type=int, default=16,
                       help="Number of frames per video segment")
    parser.add_argument('--num_workers', type=int, default=16,
                       help="Number of dataloader workers")
    parser.add_argument('--num_gen', type=int, default=None,
                       help="Max number of generated videos to evaluate")
    parser.add_argument('--num_gt', type=int, default=None,
                       help="Max number of GT videos to evaluate")
    parser.add_argument('--write_metric_to', type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device: cuda or cpu")

    args = parser.parse_args()

    # Calculate metrics
    res_all = calculate_video_metrics(
        root_dir=args.root_dir,
        path_gen=args.path_gen,
        path_gt=args.path_gt,
        metrics=args.type,
        batch_size=args.batch_size,
        sample_duration=args.sample_duration,
        num_gen=args.num_gen,
        num_gt=args.num_gt,
        num_workers=args.num_workers,
        device=args.device
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for metric, score in res_all.items():
        print(f"{metric}: {score:.4f}")
    
    # Save results
    if args.write_metric_to is not None:
        os.makedirs(os.path.dirname(args.write_metric_to), exist_ok=True)
        with open(args.write_metric_to, 'w') as f:
            json.dump(res_all, f, indent=2)
        print(f"\nResults saved to: {args.write_metric_to}")