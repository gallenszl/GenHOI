"""
Standalone FVD (Fréchet Video Distance) computation module.
Extracted from DisCo project for easier integration.
"""

from .fvd import (
    compute_fvd,
    compute_fvd_from_dirs,
    frechet_distance,
    fid_from_feats,
    build_feature_extractor,
    DatasetFVDVideoResize,
)

__all__ = [
    'compute_fvd',
    'compute_fvd_from_dirs',
    'frechet_distance',
    'fid_from_feats',
    'build_feature_extractor',
    'DatasetFVDVideoResize',
]