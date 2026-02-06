# Video Metrics Module

This module provides FID-VID and FVD evaluation metrics for video generation tasks.

## Overview

This is a self-contained implementation extracted from DisCo's `metric_center.py`, designed to make GenHOI independent and ready for open-source release.

### Supported Metrics

- **FID-VID (FVD-3DRN50)**: Video-level FID using 3D ResNet50 features
- **FVD (FVD-3DInception)**: Fréchet Video Distance using 3D Inception features

## Installation

### Prerequisites

```bash
pip install torch torchvision numpy scipy tqdm ffmpeg-python pillow
```

### Model Weights

Download the required pretrained model weights:

1. **ResNet50 Kinetics** (for FID-VID):
   - Download `resnet-50-kinetics.pth` 
   - Place in `tools/eval_fvd/resnet-50-kinetics.pth`

2. **I3D Pretrained** (for FVD):
   - Download `i3d_pretrained_400.pt`
   - Place in `tools/eval_fvd/i3d_pretrained_400.pt`

## Usage

### As a Python Module

```python
from video_metrics import calculate_video_metrics

results = calculate_video_metrics(
    root_dir="/path/to/data",
    path_gen="generated_videos/",
    path_gt="ground_truth_videos/",
    metrics=['fid-vid', 'fvd'],
    sample_duration=16,
    device='cuda'
)

print(results)
# Output: {'FVD-3DRN50': 123.45, 'FVD-3DInception': 234.56}
```

### Command Line Interface

```bash
python tools/video_metrics/metric_calculator.py \
    --root_dir /path/to/data \
    --path_gen generated_videos/ \
    --path_gt ground_truth_videos/ \
    --type fid-vid fvd \
    --sample_duration 16 \
    --write_metric_to results.json
```

### Integrated in batch_eval.sh

The module is automatically called by `batch_eval.sh`:

```bash
bash tools/batch_eval.sh /path/to/results 401 401 cuda
```

## Module Structure

```
video_metrics/
├── __init__.py              # Module exports
├── metric_calculator.py     # Main evaluation logic
├── video_dataset.py         # Video data loaders
├── feature_extractor.py     # Model builder
├── resizer.py               # Image resizing utilities
├── models/
│   ├── __init__.py
│   ├── resnet3d.py          # 3D ResNet50 model
│   └── inception3d.py       # 3D Inception model
└── README.md                # This file
```

## API Reference

### `calculate_video_metrics()`

Main function for computing video metrics.

**Parameters:**
- `root_dir` (str): Root directory for relative paths
- `path_gen` (str): Path to generated videos
- `path_gt` (str): Path to ground truth videos
- `metrics` (list): List of metrics to compute, e.g., `['fid-vid', 'fvd']`
- `sample_duration` (int, default=16): Number of frames per video segment
- `batch_size` (int, default=1): Batch size (should be 1 for video metrics)
- `num_workers` (int, default=16): Number of dataloader workers
- `device` (str, default='cuda'): Device to use ('cuda' or 'cpu')

**Returns:**
- Dictionary with metric results, e.g., `{'FVD-3DRN50': 123.45, 'FVD-3DInception': 234.56}`

## Supported Input Formats

- **Video files**: `.mp4`, `.gif`
- **Frame sequences**: `.jpg`, `.png` (with specific naming patterns)

## Technical Details

### FID-VID (FVD-3DRN50)

- Uses 3D ResNet50 pretrained on Kinetics-400
- Input size: 112×112 pixels
- Extracts features from video segments
- Computes Fréchet distance between generated and GT distributions

### FVD (FVD-3DInception)

- Uses 3D Inception (I3D) pretrained on Kinetics-400
- Input size: 224×224 pixels
- Similar feature extraction and distance computation

### Video Processing

1. Videos are decoded frame-by-frame
2. Each frame is resized using bicubic interpolation
3. Frames are grouped into segments of `sample_duration` frames
4. Features are extracted per segment and averaged per video
5. Fréchet distance is computed between distributions

## Differences from Original DisCo Implementation

1. **Simplified**: Removed unused metrics (FID-Img, IS, SSIM, LPIPS, etc.)
2. **Self-contained**: No external dependencies on DisCo codebase
3. **Cleaner API**: More intuitive function signatures
4. **Better documentation**: Comprehensive comments and docstrings

## Performance

- FID-VID computation: ~0.5s per video (GPU)
- FVD computation: ~1.0s per video (GPU)
- Memory usage: ~2GB GPU memory for batch_size=1

## Troubleshooting

### FileNotFoundError: Model weights not found

Make sure you've downloaded and placed the model weights:
- `tools/eval_fvd/resnet-50-kinetics.pth`
- `tools/eval_fvd/i3d_pretrained_400.pt`

### CUDA out of memory

Try using CPU instead:
```bash
python tools/video_metrics/metric_calculator.py ... --device cpu
```

### ffmpeg errors

Ensure ffmpeg is installed:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Citation

If you use this module, please cite:

```bibtex
@article{disco2024,
  title={DisCo: Disentangled Control for Referring Human Dance Generation in Real World},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This module is part of GenHOI and follows the same license.