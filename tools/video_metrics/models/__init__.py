"""
3D Neural Network Models for Video Feature Extraction
"""

from .resnet3d import resnet50
from .inception3d import InceptionI3d

__all__ = ['resnet50', 'InceptionI3d']
