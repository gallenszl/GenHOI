"""
GenHOI Video Metrics Module
Provides FID-VID and FVD evaluation for video generation tasks.

This module is extracted from DisCo's metric_center.py to make GenHOI
self-contained and ready for open-source release.
"""

from .metric_calculator import calculate_video_metrics

__version__ = "1.0.0"
__all__ = ['calculate_video_metrics']