#!/usr/bin/env python3
"""
GenHOI Model Download Script

This script downloads all required model weights from Hugging Face or Google Drive.

Usage:
    python scripts/download_models.py --source huggingface
    python scripts/download_models.py --source gdrive
    python scripts/download_models.py --source huggingface --models base  # only base model
    python scripts/download_models.py --source huggingface --models genhoi  # only GenHOI weights
    python scripts/download_models.py --source huggingface --models eval  # only eval models
"""

import argparse
import os
import sys
from pathlib import Path

# ============================================
# Configuration - Update these after uploading
# ============================================

# Hugging Face repository
HF_REPO_ID = "your-username/GenHOI"  # TODO: Update this

# Google Drive file IDs (extract from sharing links)
GDRIVE_IDS = {
    "GenHOI_wan_flf.consolidated": "your-file-id-here",  # TODO: Update this
    "i3d_pretrained_400.pt": "your-file-id-here",  # TODO: Update this
    "resnet-50-kinetics.pth": "your-file-id-here",  # TODO: Update this
}

# Google Drive folder ID for Wan2.1 base model
GDRIVE_WAN21_FOLDER_ID = "your-folder-id-here"  # TODO: Update this


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import gdown
    except ImportError:
        missing.append("gdown")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install them with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_from_huggingface(repo_id, local_dir, patterns=None):
    """Download files from Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    print(f"\n📥 Downloading from Hugging Face: {repo_id}")
    print(f"   Target directory: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    if patterns:
        for pattern in patterns:
            print(f"   Downloading pattern: {pattern}")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=pattern, local_dir_use_symlinks=False)
    else:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("✅ Download complete!")


def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive."""
    import gdown
    print(f"\n📥 Downloading from Google Drive...")
    print(f"   File ID: {file_id}")
    print(f"   Output: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    print("✅ Download complete!")


def get_project_root():
    """Get the project root directory."""
    script_dir = Path(__file__).parent.absolute()
    return script_dir.parent


def main():
    parser = argparse.ArgumentParser(description="Download GenHOI model weights")
    parser.add_argument("--source", type=str, choices=["huggingface", "gdrive"], default="huggingface")
    parser.add_argument("--models", type=str, choices=["all", "base", "genhoi", "eval", "demo"], default="all")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=HF_REPO_ID)
    args = parser.parse_args()
    
    check_dependencies()
    output_dir = Path(args.output_dir) if args.output_dir else get_project_root()
    models_dir = output_dir / "models"
    tools_dir = output_dir / "tools" / "eval_fvd"
    
    print("=" * 60)
    print("GenHOI Model Downloader")
    print("=" * 60)
    print(f"Source: {args.source}, Models: {args.models}, Output: {output_dir}")
    
    if args.source == "huggingface":
        if args.models == "all":
            download_from_huggingface(args.repo_id, str(models_dir))
        elif args.models == "base":
            download_from_huggingface(args.repo_id, str(models_dir), patterns=["Wan2.1-I2V-14B-720P/*"])
        elif args.models == "genhoi":
            download_from_huggingface(args.repo_id, str(models_dir), patterns=["*.consolidated"])
        elif args.models == "eval":
            os.makedirs(tools_dir, exist_ok=True)
            download_from_huggingface(args.repo_id, str(tools_dir), patterns=["eval_models/*"])
    elif args.source == "gdrive":
        if args.models in ["all", "genhoi"] and GDRIVE_IDS.get("GenHOI_wan_flf.consolidated") != "your-file-id-here":
            download_from_gdrive(GDRIVE_IDS["GenHOI_wan_flf.consolidated"], str(models_dir / "GenHOI_wan_flf.consolidated"))
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()
