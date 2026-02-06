# GenHOI 开源上传指南

本文档说明如何将 GenHOI 项目的大文件上传到 Hugging Face 或 Google Drive，以便其他用户下载使用。

## 📦 需要上传的文件清单

根据 `.gitignore` 文件，以下内容需要单独上传到云存储：

### 1. 模型权重 (Models)

| 文件/目录 | 大小 (估计) | 说明 | 上传位置 |
|-----------|-------------|------|----------|
| `models/Wan2.1-I2V-14B-720P/` | ~28GB | Wan2.1 基础模型 | Hugging Face |
| `models/GenHOI_wan_flf.consolidated` | ~2-5GB | GenHOI 微调权重 | Hugging Face |

**Wan2.1-I2V-14B-720P 目录内容：**
```
models/Wan2.1-I2V-14B-720P/
├── diffusion_pytorch_model-00001-of-00007.safetensors
├── diffusion_pytorch_model-00002-of-00007.safetensors
├── diffusion_pytorch_model-00003-of-00007.safetensors
├── diffusion_pytorch_model-00004-of-00007.safetensors
├── diffusion_pytorch_model-00005-of-00007.safetensors
├── diffusion_pytorch_model-00006-of-00007.safetensors
├── diffusion_pytorch_model-00007-of-00007.safetensors
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── Wan2.1_VAE.pth
```

### 2. 评估模型权重 (Evaluation Models)

| 文件 | 大小 (估计) | 说明 | 上传位置 |
|------|-------------|------|----------|
| `tools/eval_fvd/i3d_pretrained_400.pt` | ~50MB | I3D 预训练模型 (FVD 计算) | Hugging Face |
| `tools/eval_fvd/resnet-50-kinetics.pth` | ~100MB | ResNet-50 Kinetics 预训练 (FVD 计算) | Hugging Face |

### 3. Demo 数据 (Demo Data)

| 目录 | 大小 (估计) | 说明 | 上传位置 |
|------|-------------|------|----------|
| `demo/` | ~100-500MB | Demo 视频和参考图像 | Hugging Face / Google Drive |

**Demo 目录结构：**
```
demo/
├── 10/
│   └── 26_78/
│       ├── video.mp4           # 原始视频
│       ├── mask.mp4            # 物体掩码视频
│       ├── video_replace.mp4   # 替换后的视频
│       ├── ref_img.png         # 参考图像
│       └── 0.png, 80.png, ...  # 关键帧
├── selfswap/
│   ├── demo.csv
│   ├── demo_selfswap.csv
│   └── 10/
│       ├── video_cut/
│       ├── obj_mask_cut/
│       ├── masked_object_cut_/
│       └── object_mask_cut_/
└── demo.csv
```

### 4. 测试数据集 (Test Data) [可选]

| 目录 | 说明 | 上传位置 |
|------|------|----------|
| `data/long_video_swap/` | Object Swap 测试集 | Google Drive |
| `data/AnchorCrafter-400_405f/` | Self-Swap 测试集 (AnchorCrafter) | Google Drive |

---

## 🚀 Hugging Face 上传指南

### Step 1: 创建 Hugging Face 账号和仓库

1. 注册 [Hugging Face](https://huggingface.co/) 账号
2. 创建新的 Model Repository：
   - 主仓库名: `your-username/GenHOI`
   - 可选: 单独的数据仓库 `your-username/GenHOI-demo-data`

### Step 2: 安装 Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 3: 上传模型权重

**方法一：使用 huggingface-cli（推荐大文件）**

```bash
# 创建仓库
huggingface-cli repo create GenHOI --type model

# 克隆仓库
git lfs install
git clone https://huggingface.co/your-username/GenHOI
cd GenHOI

# 复制模型文件
cp -r /path/to/GenHOI/models/Wan2.1-I2V-14B-720P ./Wan2.1-I2V-14B-720P
cp /path/to/GenHOI/models/GenHOI_wan_flf.consolidated ./

# 复制评估模型
mkdir -p eval_models
cp /path/to/GenHOI/tools/eval_fvd/i3d_pretrained_400.pt ./eval_models/
cp /path/to/GenHOI/tools/eval_fvd/resnet-50-kinetics.pth ./eval_models/

# 上传
git add .
git commit -m "Add GenHOI model weights"
git push
```

**方法二：使用 Python API**

```python
from huggingface_hub import HfApi, upload_folder

api = HfApi()

# 上传整个目录
upload_folder(
    folder_path="/path/to/GenHOI/models/Wan2.1-I2V-14B-720P",
    repo_id="your-username/GenHOI",
    path_in_repo="Wan2.1-I2V-14B-720P",
    repo_type="model"
)

# 上传单个文件
api.upload_file(
    path_or_fileobj="/path/to/GenHOI/models/GenHOI_wan_flf.consolidated",
    path_in_repo="GenHOI_wan_flf.consolidated",
    repo_id="your-username/GenHOI",
    repo_type="model"
)
```

### Step 4: 创建 Model Card

在 Hugging Face 仓库根目录创建 `README.md`（Model Card）：

```markdown
---
license: apache-2.0
tags:
  - video-generation
  - human-object-interaction
  - wan2.1
  - diffusion
language:
  - en
pipeline_tag: text-to-video
---

# GenHOI: Generalizable Human-Object Interaction Video Generation

## Model Description

GenHOI is a generalizable framework for generating realistic human-object interaction videos.

## Files

- `Wan2.1-I2V-14B-720P/`: Base Wan2.1 model weights
- `GenHOI_wan_flf.consolidated`: Fine-tuned GenHOI weights
- `eval_models/`: Evaluation model weights (I3D, ResNet-50)

## Usage

See [GitHub Repository](https://github.com/your-username/GenHOI) for detailed instructions.

## License

Apache 2.0
```

---

## ☁️ Google Drive 上传指南

### Step 1: 创建文件夹结构

在 Google Drive 中创建以下文件夹：

```
GenHOI/
├── models/
│   ├── Wan2.1-I2V-14B-720P/
│   └── GenHOI_wan_flf.consolidated
├── eval_models/
│   ├── i3d_pretrained_400.pt
│   └── resnet-50-kinetics.pth
├── demo/
│   └── (demo 数据)
└── data/
    ├── long_video_swap/
    └── AnchorCrafter-400_405f/
```

### Step 2: 上传文件

1. 打开 [Google Drive](https://drive.google.com/)
2. 创建 `GenHOI` 文件夹
3. 按上述结构上传文件

### Step 3: 设置共享权限

1. 右键点击 `GenHOI` 文件夹
2. 选择 "共享" -> "获取链接"
3. 设置为 "知道链接的任何人都可以查看"
4. 复制共享链接

### Step 4: 创建下载脚本

创建 `download_from_gdrive.py`：

```python
import gdown
import os

# Google Drive 文件夹 ID（从共享链接中提取）
GDRIVE_FOLDER_ID = "your-folder-id-here"

# 各文件的下载 ID
FILES = {
    "models/GenHOI_wan_flf.consolidated": "file-id-1",
    "eval_models/i3d_pretrained_400.pt": "file-id-2",
    "eval_models/resnet-50-kinetics.pth": "file-id-3",
}

def download_file(file_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

if __name__ == "__main__":
    for path, file_id in FILES.items():
        print(f"Downloading {path}...")
        download_file(file_id, path)
```

---

## 📝 更新 README.md

上传完成后，更新项目 README.md 中的下载链接：

```markdown
## 📂 Model Weights

### Option 1: Hugging Face (Recommended)

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download all models
huggingface-cli download your-username/GenHOI --local-dir models/
```

### Option 2: Google Drive

Download from: [Google Drive Link](https://drive.google.com/drive/folders/xxx)

Or use the download script:
```bash
pip install gdown
python scripts/download_from_gdrive.py
```
```

---

## ✅ 上传检查清单

- [ ] **模型权重**
  - [ ] Wan2.1-I2V-14B-720P (7个 safetensors + 3个 pth)
  - [ ] GenHOI_wan_flf.consolidated
  
- [ ] **评估模型**
  - [ ] i3d_pretrained_400.pt
  - [ ] resnet-50-kinetics.pth

- [ ] **Demo 数据**
  - [ ] demo/10/26_78/ (视频、掩码、参考图)
  - [ ] demo/selfswap/ (selfswap demo 数据)
  - [ ] demo.csv, demo_selfswap.csv

- [ ] **测试数据** (可选)
  - [ ] data/long_video_swap/
  - [ ] data/AnchorCrafter-400_405f/

- [ ] **文档更新**
  - [ ] 更新 README.md 下载链接
  - [ ] 创建 Hugging Face Model Card
  - [ ] 验证下载脚本可用

---

## 🔗 推荐的仓库结构

### Hugging Face

```
your-username/GenHOI (Model Repository)
├── README.md (Model Card)
├── Wan2.1-I2V-14B-720P/
│   ├── diffusion_pytorch_model-*.safetensors
│   ├── models_clip_*.pth
│   ├── models_t5_*.pth
│   └── Wan2.1_VAE.pth
├── GenHOI_wan_flf.consolidated
└── eval_models/
    ├── i3d_pretrained_400.pt
    └── resnet-50-kinetics.pth
```

### GitHub

```
your-username/GenHOI (Code Repository)
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── diffsynth/
├── examples/
├── tools/
├── assets/
└── scripts/
    └── download_models.py
```

---

## 📧 联系方式

如有上传问题，请联系项目维护者或在 GitHub Issues 中提问。