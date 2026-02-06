# GenHOI Evaluation Tools

This directory contains evaluation scripts and tools for assessing video generation quality.

## Directory Structure

```
tools/
├── batch_eval.sh           # Main batch evaluation script
├── batch_oc_eval.py        # Object-CLIP (OC) metric evaluation
├── oc_metric_with_viz.py   # OC metric implementation with visualization
├── Object_CLIP.py          # Object-CLIP helper functions
├── clip_similarity.py      # CLIP similarity utilities
├── psnr.py                 # Per-video PSNR calculation
├── fid.py                  # FID, FID-VID, PSNR, SSIM evaluation
├── inception.py            # Inception model for FID
└── README.md               # This file
```

## Quick Start

### Basic Usage

```bash
# Run full evaluation on a results directory
bash tools/batch_eval.sh /path/to/results 401 401 cuda
```

### Parameters

- `base_dir`: Directory containing generated videos and ground truth
- `sample_frames`: Number of sample frames (default: 401)
- `duration`: Sample duration (default: 401)
- `device`: Device to use - `cuda` or `cpu` (default: cuda)

### Example

```bash
bash tools/batch_eval.sh /path/to/my_experiment 81 81 cuda
```

## Required Directory Structure

Your `base_dir` should contain subdirectories with the following structure:

```
base_dir/
├── sample_0000_allclips/
│   ├── all_generated.mp4   # Generated video
│   ├── all_gt.mp4          # Ground truth video
│   ├── all_handpose.mp4    # Hand pose / bbox mask video
│   └── all_ref.mp4         # Reference frame/video
├── sample_0001_allclips/
│   └── ...
└── ...
```

## Evaluation Metrics

The script computes the following metrics:

1. **Object-CLIP (OC)**: Measures object consistency using CLIP embeddings
   - Output: `base_dir/oc_scores.csv`
   - Visualization: `base_dir/oc_vis/`

2. **FID-VID**: Fréchet Inception Distance for videos (using 3D ResNet50)
   - Output: `base_dir_gen/metrics_fid-vid_fvd.json`

3. **FVD**: Fréchet Video Distance (using 3D Inception)
   - Output: `base_dir_gen/metrics_fid-vid_fvd.json`

4. **FID**: Fréchet Inception Distance for frames
   - Output: `base_dir_all/evaluation_results.csv`

5. **PSNR**: Peak Signal-to-Noise Ratio
   - Output: `base_dir_all/evaluation_results.csv`

6. **SSIM**: Structural Similarity Index
   - Output: `base_dir_all/evaluation_results.csv`

7. **All Metrics Merged**: Combined results from all evaluations
   - Output: `base_dir/all_metrics.json`

## Individual Tool Usage

### 1. Object-CLIP Evaluation

```bash
python tools/batch_oc_eval.py \
    --root /path/to/base_dir \
    --stride 1 \
    --aggregate mean \
    --device cuda
```

### 2. FID + PSNR + SSIM

```bash
python tools/fid.py \
    --video_dir /path/to/videos \
    --device cuda:0
```

### 3. Per-Video PSNR

```bash
python tools/psnr.py \
    --input_dir /path/to/base_dir \
    --output_csv /path/to/output.csv
```

## Configuration

You can set the maximum number of samples to evaluate by setting the `MAX_SAMPLES` environment variable:

```bash
# Evaluate all samples
bash tools/batch_eval.sh /path/to/results

# Evaluate only first 8 samples
MAX_SAMPLES=8 bash tools/batch_eval.sh /path/to/results
```

## Dependencies

Required Python packages:
- torch
- torchvision
- numpy
- PIL/Pillow
- open-clip-torch or clip
- decord or opencv-python
- imageio
- scipy
- scikit-image
- tqdm

## Output Files

After running the evaluation, you will find:

```
base_dir/
├── oc_scores.csv              # OC metric results
├── oc_vis/                    # OC visualizations
├── psnr_per_video.csv         # Per-video PSNR
├── sample_*_generated.mp4     # Organized generated videos
├── sample_*_gt.mp4            # Organized ground truth
├── sample_*_handpose.mp4      # Organized bbox masks
└── sample_*_ref.mp4           # Organized references

base_dir_gen/
├── metrics_fid-vid_fvd.json   # FID-VID and FVD results
└── sample_*_generated.mp4     # Generated videos

base_dir_gt/
└── sample_*_gt.mp4            # Ground truth videos

base_dir_all/
├── evaluation_results.csv     # FID, PSNR, SSIM results
├── sample_*_generated.mp4     # All generated videos
└── sample_*_gt.mp4            # All ground truth videos
```

## Notes

- The script automatically organizes files into `_gen`, `_gt`, and `_all` subdirectories
- OC visualization videos show bounding boxes and scores overlaid on generated videos
- FID-VID and FVD require the external `metric_center.py` script (optional)
- All metrics are computed on GPU by default for faster processing

## Troubleshooting

**Issue**: `metric_center.py` not found
- This is optional. The script will skip FID-VID and FVD if not available.

**Issue**: CUDA out of memory
- Use `device=cpu` parameter
- Reduce batch size in the scripts

**Issue**: Missing video files
- Ensure your directory structure matches the required format
- Check that video files are named correctly (`all_generated.mp4`, etc.)

## License

See the main project LICENSE file.