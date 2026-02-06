#!/usr/bin/env python3
"""
Unified results merger - reads all metric files and combines into one JSON.
No intermediate directories needed.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path


def read_video_metrics_json(json_path):
    """Read FID-VID and FVD from JSON."""
    results = {}
    if not os.path.exists(json_path):
        return results
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                results[key] = value
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}")
    
    return results


def read_frame_metrics_csv(csv_path):
    """Read FID, PSNR, SSIM from CSV."""
    results = {}
    if not os.path.exists(csv_path):
        return results
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['Metric']
                value = float(row['Value']) if row['Value'] else None
                results[metric] = value
    except Exception as e:
        print(f"Warning: Failed to read {csv_path}: {e}")
    
    return results


def read_oc_scores(oc_csv_path):
    """Read Object-CLIP scores from CSV."""
    results = {}
    if not os.path.exists(oc_csv_path):
        return results
    
    try:
        with open(oc_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['sample'] == '__OVERALL_MEAN__':
                    results['OC_overall_mean'] = float(row['overall']) if row['overall'] else None
    except Exception as e:
        print(f"Warning: Failed to read {oc_csv_path}: {e}")
    
    return results


def merge_all_results_unified(base_dir):
    """
    Merge all evaluation results from base_dir (no subdirectories).
    
    Args:
        base_dir: Base directory containing all result files
    
    Returns:
        Dictionary with all metrics
    """
    all_results = {}
    
    # 1. Read video metrics (FID-VID, FVD)
    video_metrics_path = os.path.join(base_dir, 'metrics_video.json')
    video_results = read_video_metrics_json(video_metrics_path)
    all_results.update(video_results)
    
    # 2. Read frame metrics (FID, PSNR, SSIM)
    frame_metrics_path = os.path.join(base_dir, 'metrics_frame.csv')
    frame_results = read_frame_metrics_csv(frame_metrics_path)
    all_results.update(frame_results)
    
    # 3. Read OC scores (optional)
    oc_path = os.path.join(base_dir, 'oc_scores.csv')
    oc_results = read_oc_scores(oc_path)
    all_results.update(oc_results)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Merge all evaluation results into a single unified JSON file"
    )
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base directory containing evaluation results")
    parser.add_argument('--output', type=str, required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Merge results
    all_results = merge_all_results_unified(args.base_dir)
    
    if not all_results:
        print("Warning: No results found!")
        all_results = {"error": "No metrics computed"}
    
    # Write output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("UNIFIED RESULTS")
    print("="*70)
    for metric, value in all_results.items():
        if value is not None and isinstance(value, (int, float)):
            print(f"{metric:30s}: {value:.4f}")
        else:
            print(f"{metric:30s}: {value}")
    print("="*70)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()