# GenHOI-VACE

<p align="center">
  <img src="src/teaser.png" width="100%">
</p>

GenHOI-VACE 是一个基于 [Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) 的人物-物体交互（Human-Object Interaction）视频生成框架。本项目通过 Gate Attention 机制和 LoRA 微调，实现高质量的 HOI 视频生成与编辑。

## 🌟 特性

- **高质量 HOI 视频生成**：基于 Wan2.1-VACE-14B 强大的视频生成能力
- **Gate Attention 机制**：通过可学习的门控注意力增强物体交互建模
- **LoRA 微调**：高效的参数微调策略，支持快速适配不同场景
- **多 GPU 并行推理**：支持多卡并行，大幅提升推理效率
- **长视频生成**：支持首尾帧续推（First-Last Frame）模式，生成任意长度视频
- **灵活的数据格式**：支持多种数据集格式（AnchorCrafter、通用 HOI 格式）

## 📋 目录

- [环境配置](#-环境配置)
- [模型权重](#-模型权重)
- [快速开始](#-快速开始)
- [数据集格式](#-数据集格式)
- [推理示例](#-推理示例)
- [项目结构](#-项目结构)
- [致谢](#-致谢)

## 🔧 环境配置

### 依赖安装

```bash
# 创建 conda 环境
conda create -n genhoi python=3.10 -y
conda activate genhoi

# 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 主要依赖

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.1
- decord
- einops
- modelscope
- safetensors

## 📦 模型权重

### 基础模型

从 HuggingFace 下载 Wan2.1-VACE-14B 基础模型：

```bash
# 使用 modelscope 或 huggingface-cli 下载
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/Wan2.1-VACE-14B
```

### GenHOI 权重

将 GenHOI 微调权重放置在 `models/GenHOI_VACE/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `step-5000-gate_attn.safetensors` | Gate Attention 权重 |
| `step-1700-lora-gate-720.safetensors` | LoRA 权重 (720p) |
| `step-1100-lora-gate-flf-720-2.safetensors` | First-Last Frame 模式 LoRA 权重 |

## 🚀 快速开始

### Self-Swap 推理（自身替换）

适用于 AnchorCrafter 风格的数据集，保持人物外观一致性：

```bash
python examples/wanvideo/selfswap_infer.py \
    --output_dir results/selfswap_demo \
    --data_csv demo/demo.csv \
    --gpus 0 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors
```

### Object-Swap 推理（物体替换）

适用于 HOI 物体交换任务：

```bash
python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_demo \
    --data_csv demo/demo.csv \
    --gpus 0 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors
```

### 启用 First-Last Frame 模式（长视频生成）

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

## 📁 数据集格式

数据集使用 CSV 格式，包含以下字段：

```csv
video_path,object_mask_path,hand_mask_path,depth_path,prompt,reference_image
/path/to/video.mp4,/path/to/object_mask.mp4,/path/to/hand_mask.mp4,/path/to/depth.mp4,"A person holding an object",/path/to/ref.jpg
```

### 字段说明

| 字段 | 说明 | 必需 |
|------|------|------|
| `video_path` | 原始视频路径 | ✅ |
| `object_mask_path` | 物体 mask 视频路径 | ✅ |
| `hand_mask_path` | 手部 mask 视频路径 | 可选 |
| `depth_path` | 深度图视频路径 | 可选 |
| `prompt` | 文本描述 | ✅ |
| `reference_image` | 参考图像路径 | ✅ |

## 🎯 推理示例

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_dir` | 输出目录 | `results/` |
| `--data_csv` | 数据集 CSV 文件路径 | - |
| `--gpus` | 使用的 GPU 索引，逗号分隔 | `0,1,2,3` |
| `--max_num_frames` | 最大帧数 | `81` |
| `--is_fl` | 启用 First-Last Frame 模式 | `False` |
| `--model_path` | Gate Attention 模型路径 | - |
| `--lora_path` | LoRA 权重路径 | - |

### 多 GPU 并行推理

```bash
# 使用 4 张 GPU 并行推理
python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_multi_gpu \
    --data_csv data/dataset.csv \
    --gpus 0,1,2,3 \
    --max_num_frames 81
```

### 长视频生成（401 帧）

```bash
python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_long \
    --data_csv data/dataset.csv \
    --gpus 0,1,2,3 \
    --max_num_frames 401 \
    --is_fl
```

## 📂 项目结构

```
GenHOI/
├── diffsynth/                    # 核心推理库
│   ├── models/                   # 模型定义
│   │   ├── wan_video_vace.py     # VACE 模型
│   │   ├── wan_video_dit.py      # DiT 模型
│   │   ├── wan_video_vae.py      # VAE 模型
│   │   └── set_condition_branch.py # Gate Attention 设置
│   ├── pipelines/                # 推理 Pipeline
│   │   └── wan_video_new.py      # Wan Video Pipeline
│   ├── prompters/                # Prompt 处理
│   └── schedulers/               # 调度器
├── examples/
│   └── wanvideo/
│       ├── selfswap_infer.py     # Self-Swap 推理脚本
│       ├── swap_infer.py         # Object-Swap 推理脚本
│       └── dataset/              # 数据集处理
│           ├── customer_dataset.py
│           └── customer_dataset_anchorcrafter.py
├── models/
│   └── GenHOI_VACE/              # 模型权重
├── results/                      # 推理结果
├── demo/                         # Demo 数据
└── README.md
```

## 🔍 技术细节

### Gate Attention 机制

GenHOI 通过在 VACE 模块中引入可学习的 Gate Attention，增强对人物-物体交互的建模能力：

```python
from diffsynth.models.set_condition_branch import set_stand_in

# 初始化 Gate Attention
set_stand_in(
    pipe.vace,
    model_path=None,
    train=False,
    only_gate=True
)
```

### 首尾帧续推

长视频通过分段生成并使用首尾帧续推策略保持时序一致性：

1. 将长视频分割为多个 clip（每个 81 帧）
2. 每个 clip 的首帧使用上一个 clip 的最后一帧
3. 对应的 mask 首帧置为全黑（表示该区域已生成）

## 🙏 致谢

本项目基于以下开源工作：

- [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) - 基础视频生成模型
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - 推理框架
- [AnchorCrafter](https://github.com/AnchorCrafter/AnchorCrafter) - 数据集格式参考

## 📄 License

本项目遵循 Apache 2.0 License。

## ?? 联系方式

如有问题或建议，欢迎提交 Issue 或 PR。