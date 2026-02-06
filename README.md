# GenHOI: Generalizable Human-Object Interaction Video Generation

<p align="center">
  <img src="assets/teaser.png" width="90%">
</p>

GenHOI is a generalizable framework for generating realistic human-object interaction videos. Built upon the Wan2.1 video generation model and DiffSynth-Studio, GenHOI enables flexible object swapping and HOI video synthesis with fine-grained control.

## ✨ Features

- **Generalizable Object Swapping**: Replace objects in videos while maintaining natural hand-object interactions
- **High-Quality Video Generation**: Based on Wan2.1-I2V-14B model for photorealistic results
- **Flexible Frame Control**: Support for variable frame lengths (up to 400+ frames)
- **Multi-GPU Inference**: Distributed processing for efficient generation
- **Fine-grained Control**: Object mask and reference image guided generation

## 📦 Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 12.x
- PyTorch >= 2.0

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/GenHOI.git
cd GenHOI

# Create virtual environment (recommended)
conda create -n genhoi python=3.10 -y
conda activate genhoi

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 📂 Model Weights

### Option 1: Download with Script (Recommended)

```bash
# Install dependencies
pip install huggingface_hub gdown

# Download all models from Hugging Face
python scripts/download_models.py --source huggingface

# Or download specific components
python scripts/download_models.py --source huggingface --models base    # Wan2.1 base model only
python scripts/download_models.py --source huggingface --models genhoi  # GenHOI weights only
python scripts/download_models.py --source huggingface --models eval    # Evaluation models only
```

### Option 2: Manual Download from Hugging Face

Download from: 🤗 [Hugging Face - GenHOI](https://huggingface.co/your-username/GenHOI)

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download your-username/GenHOI --local-dir models/
```

### Option 3: Manual Download from Google Drive

Download from: [Google Drive - GenHOI](https://drive.google.com/drive/folders/xxx)

### Model Files Structure

After downloading, your `models/` directory should look like:

```
models/
├── Wan2.1-I2V-14B-720P/           # Base model (~28GB)
│   ├── diffusion_pytorch_model-00001-of-00007.safetensors
│   ├── diffusion_pytorch_model-00002-of-00007.safetensors
│   ├── diffusion_pytorch_model-00003-of-00007.safetensors
│   ├── diffusion_pytorch_model-00004-of-00007.safetensors
│   ├── diffusion_pytorch_model-00005-of-00007.safetensors
│   ├── diffusion_pytorch_model-00006-of-00007.safetensors
│   ├── diffusion_pytorch_model-00007-of-00007.safetensors
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1_VAE.pth
└── GenHOI_wan_flf.consolidated    # Fine-tuned weights (~2-5GB)
```

### Evaluation Models (Optional)

For running evaluation metrics (FVD, FID), download additional models to `tools/eval_fvd/`:

```
tools/eval_fvd/
├── i3d_pretrained_400.pt      # I3D model for FVD (~50MB)
└── resnet-50-kinetics.pth     # ResNet-50 Kinetics (~100MB)
```

> **Note**: Update the download links after uploading. See [UPLOAD_GUIDE.md](UPLOAD_GUIDE.md) for upload instructions.

## 🚀 Quick Start

### Run Demo

```bash
python examples/wanvideo/test_swap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/demo \
    --data_csv demo/demo.csv \
    --gpus 0 \
    --max_num_frames 81 \
    --is_fl
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `models/ckpt/first_frame_rope.consolidated` | Path to GenHOI checkpoint |
| `--output_dir` | `results/demo` | Output directory for generated videos |
| `--data_csv` | `demo/demo.csv` | Path to input data CSV file |
| `--gpus` | `0` | GPU indices (comma-separated, e.g., `0,1,2,3`) |
| `--max_num_frames` | `81` | Maximum frames to generate |
| `--is_fl` | `False` | Enable first-last frame mode |

### Multi-GPU Inference

For faster inference, use multiple GPUs:

```bash
python examples/wanvideo/test_swap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/demo \
    --data_csv demo/demo.csv \
    --gpus 0,1,2,3 \
    --max_num_frames 401 \
    --is_fl
```

## 📈 Evaluation on Test Set

### Object Swap Task

Run evaluation on the object swap test set:

```bash
# 81 frames (short video)
python examples/wanvideo/test_swap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/swap_81 \
    --data_csv data/long_video_swap/swap.csv \
    --gpus 2 \
    --max_num_frames 81 \
    --is_fl

# 401 frames (long video)
python examples/wanvideo/test_swap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/swap_401 \
    --data_csv data/long_video_swap/swap_f16.csv \
    --gpus 3 \
    --max_num_frames 401
```

### Self-Swap Task

Run evaluation on the self-swap test set (AnchorCrafter benchmark):

```bash
# 81 frames (short video)
python examples/wanvideo/test_selfswap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/selfswap_81 \
    --data_csv data/AnchorCrafter-400_405f/dataset_select_f50.csv \
    --gpus 2 \
    --max_num_frames 81 \
    --is_fl

# 401 frames (long video)
python examples/wanvideo/test_selfswap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/selfswap_401 \
    --data_csv data/AnchorCrafter-400_405f/dataset_select_f16.csv \
    --gpus 3 \
    --max_num_frames 401
```

### Evaluate Results

After running inference, use the unified evaluation script to compute metrics:

```bash
# Usage: bash tools/batch_eval_unified.sh <base_dir> [sample_duration] [device]

# Evaluate 81-frame results
bash tools/batch_eval_unified.sh results/swap_81 81 cuda
bash tools/batch_eval_unified.sh results/selfswap_81 81 cuda

# Evaluate 401-frame results
bash tools/batch_eval_unified.sh results/swap_401 401 cuda
bash tools/batch_eval_unified.sh results/selfswap_401 401 cuda
```

The evaluation script computes the following metrics:

| Metric | Description |
|--------|-------------|
| **FVD** | Fréchet Video Distance (using 3D-ResNet50 and 3D-Inception) |
| **FID-VID** | Fréchet Inception Distance for video frames |
| **FID** | Fréchet Inception Distance (frame-level) |
| **PSNR** | Peak Signal-to-Noise Ratio |
| **SSIM** | Structural Similarity Index |
| **OC** | Object-CLIP similarity score |

Results are saved to `<base_dir>/all_metrics.json`.

## 📊 Data Format

### Input CSV Structure

Create a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `video_path` | Path to source video |
| `obj_mask_path` | Path to object mask video (white mask on object region) |
| `input_path` | Path to video with replaced background/object |
| `ref_img` | Path to reference image of the target object |

Example `demo.csv`:
```csv
video_path,obj_mask_path,input_path,ref_img
demo/10/26_78/video.mp4,demo/10/26_78/mask.mp4,demo/10/26_78/video_replace.mp4,demo/10/26_78/ref_img.png
```

### Preparing Your Own Data

1. **Source Video** (`video_path`): Original video with human-object interaction
2. **Object Mask** (`obj_mask_path`): Binary mask video highlighting the object region (white: object, black: background)
3. **Replacement Video** (`input_path`): Video with the original object removed/replaced
4. **Reference Image** (`ref_img`): Clear image of the target object you want to insert

## 📁 Output Structure

Generated results are saved in the specified output directory:

```
results/demo/
└── sample_0000_allclips/
    ├── all_generated.mp4      # Final generated video
    ├── all_control.mp4        # Control signal visualization
    ├── all_replaced.mp4       # Object replacement result
    ├── all_gt.mp4             # Ground truth video
    ├── all_ref.mp4            # Reference visualization
    ├── all_handpose.mp4       # Hand pose visualization
    └── all_stitched_2x2.mp4   # 2x2 grid comparison video
```

## 🔧 Advanced Configuration

### Video Resolution

The default generation resolution is **720×1280** (portrait mode). Modify the pipeline parameters in `test_swap.py` if needed:

```python
video, video_control, video_gt, video_ref, latents_hand_pose, video_replaced = pipe(
    ...
    height=1280,
    width=720,
    ...
)
```

### Inference Steps

Adjust generation quality vs. speed trade-off:

```python
num_inference_steps=50  # Higher = better quality, slower
```

## 📋 Requirements

Main dependencies:
- `torch>=2.0`
- `transformers==4.46.2`
- `controlnet-aux==0.0.7`
- `decord`
- `einops`
- `safetensors`
- `cupy-cuda12x`

See `requirements.txt` for the complete list.

## 🏗️ Project Structure

```
GenHOI/
├── diffsynth/                    # Core diffusion synthesis library
│   ├── models/                   # Model implementations
│   ├── pipelines/                # Generation pipelines
│   ├── schedulers/               # Noise schedulers
│   └── prompters/                # Text prompt processors
├── examples/
│   └── wanvideo/
│       ├── dataset/              # Dataset utilities
│       ├── test_swap.py          # Main inference script
│       └── test_selfswap.py      # Self-swap testing
├── models/                       # Model weights directory
├── demo/                         # Demo data and examples
└── requirements.txt
```

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{genhoi2025,
  title={GenHOI: Generalizable Human-Object Interaction Video Generation},
  author={},
  journal={},
  year={2025}
}
```

## 🙏 Acknowledgements

This project is built upon:
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Base video generation model
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Diffusion synthesis framework

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

## 📧 Contact

For questions and discussions, please open an issue or contact us at [your-email@example.com].

---

<p align="center">
  <b>⭐ Star us on GitHub if you find this project useful!</b>
</p>