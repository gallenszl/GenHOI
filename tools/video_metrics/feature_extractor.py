"""
Feature extractor builder for video metrics.
Extracted from DisCo's features.py
"""

import os
import sys
import torch
from pathlib import Path

# Add models directory to path
script_dir = Path(__file__).parent
models_dir = script_dir / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))


def build_feature_extractor(mode, device=torch.device("cuda"), sample_duration=16):
    """
    Build a feature extractor model for FVD computation.
    
    Args:
        mode: str, 'FVD-3DInception'
        device: torch device for model
        sample_duration: number of frames per video segment
    
    Returns:
        A feature extraction model in eval mode
    """
    # Determine model weights directory
    script_dir = Path(__file__).parent
    weights_dir = script_dir.parent / "eval_fvd"
    
    if mode == "FVD-3DInception":
        # Import and build 3D Inception
        from inception3d import InceptionI3d
        
        model = InceptionI3d(400, in_channels=3)
        
        # Load pretrained weights
        weights_path = weights_dir / "i3d_pretrained_400.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Inception3D weights not found at {weights_path}. "
                f"Please download I3D pretrained model."
            )
        
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model = model.to(device).eval()
        
    else:
        raise NotImplementedError(
            f"Mode '{mode}' is not supported. "
            f"Only 'FVD-3DInception' is implemented."
        )
    
    return model