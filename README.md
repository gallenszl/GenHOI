# GenHOI: Towards Object-Consistent Hand–Object Interaction with Temporally Balanced and Spatially Selective Object Injection

<p align="center">
  <img src="assets/teaser.png" width="90%">
</p>

This is the official repository for the paper [GenHOI: Towards Object-Consistent Hand-Object Interaction with Temporally Balanced and Spatially Selective Object Injection](https://arxiv.org/pdf/2603.06048).

<!-- ## ✨ Features

- **Generalizable Object Swapping**: Replace objects in videos while maintaining natural hand-object interactions
- **High-Quality Video Generation**: Based on Wan2.1-I2V-14B model for photorealistic results
- **Flexible Frame Control**: Support for variable frame lengths (up to 400+ frames)
- **Multi-GPU Inference**: Distributed processing for efficient generation
- **Fine-grained Control**: Object mask and reference image guided generation -->

## 🌿 Branch Information

This project contains **two branches** with different base models:

| Branch   | Base Model                                    | Description                          | Environment Reference                                                   |
| -------- | --------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| **main** | [Wan2.1](https://github.com/Wan-Video/Wan2.1) | GenHOI based on Wan2.1-I2V-14B model | [Wan2.1 Installation](https://github.com/Wan-Video/Wan2.1#installation) |
| **vace** | [VACE](https://github.com/ali-vilab/VACE)     | GenHOI based on vace model           | [VACE Installation](https://github.com/ali-vilab/VACE#installation)     |

### Switch Branch

```bash
# Use Wan2.1 based version (default)
git checkout main

# Use VACE based version
git checkout vace
```

**Below, we introduce how to start the VACE-based GenHOI. We highly recommend using the VACE-based version for better performance. For instructions on the WAN-based GenHOI, please refer to the README file in the corresponding branch.**

### Installation

Please refer to [Wan2.1-VACE](https://github.com/ali-vilab/VACE) for the environment installation instructions.

## 📦 Model Weights

### Base Model

Download Wan2.1-VACE-14B base model from HuggingFace. In most cases, the model will be downloaded automatically. If not, you can manually download it and place it in the required directory.

```bash
# Download using modelscope or huggingface-cli
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/Wan2.1-VACE-14B
```

### GenHOI Weights

Download model from: 🤗 [Hugging Face - GenHOI](https://huggingface.co/szlgallen/GenHOI)

Place GenHOI fine-tuned weights in the `models/GenHOI_VACE/` directory:

| Filename                                    | Description                        |
| ------------------------------------------- | ---------------------------------- |
| `step-5000-gate_attn.safetensors`           | Gate Attention Weights             |
| `step-1700-lora-gate-720.safetensors`       | LoRA Weights                       |
| `step-1100-lora-gate-flf-720-2.safetensors` | First-Last Frame Mode LoRA Weights |

### Model Files Structure

After downloading, your `models/` directory should look like:

```
models/
└── GenHOI_VACE/
    ├── step-5000-gate_attn.safetensors
    ├── step-1100-lora-gate-flf-720-2.safetensors
    └── step-1700-lora-gate-720.safetensors
```

<!-- ### Evaluation Models (Optional)

For running evaluation metrics (FVD, FID), download additional models to `tools/eval_fvd/`:

```
tools/eval_fvd/
├── i3d_pretrained_400.pt      # I3D model for FVD (~50MB)
└── resnet-50-kinetics.pth     # ResNet-50 Kinetics (~100MB)
``` -->

## Evaluation dataset

Please download the corresponding evaluation dataset from [Hugging Face - GenHOI-data](https://huggingface.co/datasets/szlgallen/GenHOI)

### Data Files Structure

After downloading, your `data/` directory should look like:

```
data/
├── long_video_swap/
│   ├── swap.csv
│   ├── swap_f16.csv
│   ├── 10/
│   │   ├── 0_0/
│   │   │   ├── video.mp4
│   │   │   ├── mask.mp4
│   │   │   ├── video_replace.mp4
│   │   │   ├── ref_img.png
│   │   │   ├── 0.png
│   │   │   ├── 80.png
│   │   │   └── ...
│   │   ├── 0_1/
│   │   └── ...
│   ├── 11/
│   └── 5/
│
├── AnchorCrafter-400_405f/
│   ├── dataset_select.csv
│   ├── dataset_select_f16.csv
│   ├── dataset_select_f50.csv
│   ├── 10/
│   │   ├── video_cut/
│   │   ├── obj_mask_cut/
│   │   ├── object_mask_cut_/
│   │   │   └── <clip_id>/
│   │   │       ├── 01.jpg
│   │   │       ├── 02.jpg
│   │   │       └── 03.jpg
│   │   └── masked_object_cut_/
│   │       └── <clip_id>/
│   │           ├── 01.jpg
│   │           ├── 02.jpg
│   │           └── 03.jpg
│   ├── 11/
│   ├── tune/
│   └── ...
```

### Demo Files Structure

After downloading, your `demo/` directory should look like:

```text
demo/
├── demo.csv
├── demo_selfswap.csv
├── 10/
│   └── 26_78/
│       └── ...
└── selfswap/
    └── 10/
        ├── video_cut/
        │   └── ...
        ├── obj_mask_cut/
        │   └── ...
        ├── object_mask_cut_/
        │   └── 0/
        │       └── ...
        └── masked_object_cut_/
            └── 0/
                └── ...
```

## 🚀 Quick Start

We provide a quick start demo included in this repository. To run on our full evaluation dataset, simply download the dataset from Hugging Face and change the `data_csv` argument to `data/long_video_swap/swap.csv` (from the downloaded evaluation dataset).

### Self-Swap(Reconstruct)

```bash
python examples/wanvideo/selfswap_infer.py \
    --output_dir results/selfswap_demo \
    --data_csv demo/demo_selfswap.csv \
    --gpus 0 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors
```

### Object-Swap

```bash
python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_demo \
    --data_csv demo/demo.csv \
    --gpus 0 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors
```

### Enable First-Last Frame Mode

you can just add `--is_fl` argument to enable the FLF mode

```bash
python examples/wanvideo/selfswap_infer.py \
    --output_dir results/selfswap_flf \
    --data_csv demo/demo.csv \
    --gpus 0,1,2,3 \
    --max_num_frames 401 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1100-lora-gate-flf-720-2.safetensors \
    --is_fl
```

### Evaluate Results

The evaluation code is provided in the `main` branch.

<strong style="color:red;">IMPORTANT: Please switch to the `main` branch before proceeding.</strong>

Please refer to the **Evaluation** section of the README in the `main` branch for detailed instructions.  Below, we provide a brief overview of the startup commands and supported evaluation metrics.

```bash
# Usage: bash tools/batch_eval_unified.sh <base_dir> [sample_duration] [device]

# Evaluate 81-frame results
bash tools/batch_eval_unified.sh results/swap_81 81 cuda
bash tools/batch_eval_unified.sh results/selfswap_81 81 cuda

# Evaluate 401-frame results
bash tools/batch_eval_unified.sh results/swap_401 401 cuda
bash tools/batch_eval_unified.sh results/selfswap_401 401 cuda
```

Results are saved to `<base_dir>/all_metrics.json`.

**Note:** For both **Self-Reenactment** and **Cross-Reenactment**, all the metrics listed above will be calculated.
However, for **Cross-Reenactment**, **only FVD and FID are valid metrics**.

## 📊 Data Format

### Input CSV Structure

Create a CSV file like the following example `demo.csv`:

```csv
video_path,obj_mask_path,input_path,ref_img
demo/10/26_78/video.mp4,demo/10/26_78/mask.mp4,demo/10/26_78/video_replace.mp4,demo/10/26_78/ref_img.png
```

### Preparing Your Own Data

1. **Source Video** (`video_path`): Original video with human-object interaction
2. **Object Mask** (`obj_mask_path`): Binary mask video highlighting the object region (white: object, black: background)
3. **Replacement Video** (`input_path`): Video with the original object removed/replaced
4. **Reference Image** (`ref_img`): Clear image of the target object you want to insert

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{huang2026genhoiobjectconsistenthandobjectinteraction,
      title={GenHOI: Towards Object-Consistent Hand-Object Interaction with Temporally Balanced and Spatially Selective Object Injection}, 
      author={Xuan Huang and Mochu Xiang and Zhelun Shen and Jinbo Wu and Chenming Wu and Chen Zhao and Kaisiyuan Wang and Hang Zhou and Shanshan Liu and Haocheng Feng and Wei He and Jingdong Wang},
      year={2026},
      eprint={2603.06048},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06048}, 
}
```

## 🙏 Acknowledgements

This project is based on the following open-source works:

- [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) - Base video generation model
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Inference framework

## 📄 License

This project is released under the [Creative Commons Attribution Non Commercial 4.0](LICENSE).

