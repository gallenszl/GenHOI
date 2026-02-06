#!/bin/bash
# GenHOI 完整上传脚本 - 上传所有 .gitignore 中的大文件到 HuggingFace
# 
# 使用方法:
#   后台运行: nohup bash scripts/upload_to_huggingface.sh > upload.log 2>&1 &
#   查看进度: tail -f upload.log
#
# 前提条件:
#   1. 已登录 HuggingFace: huggingface-cli login
#   2. 已安装 git-lfs: apt-get install git-lfs && git lfs install

set -e

# ============================================
# 配置部分 - 请根据实际情况修改
# ============================================
export http_proxy='http://agent.baidu.com:8891'
export https_proxy='http://agent.baidu.com:8891'

# HuggingFace 仓库配置
HF_USERNAME="XuanHuang0"
HF_DATA_REPO="${HF_USERNAME}/GenHOI-data"      # 数据仓库 (demo, assets, eval_models)
HF_MODEL_REPO="${HF_USERNAME}/GenHOI"          # 模型仓库 (GenHOI weights)

# 本地路径配置
BASE_DIR="/root/paddlejob/workspace/huangxuan/GenHOI"
LOG_FILE="${BASE_DIR}/upload_progress.log"

# ============================================
# 辅助函数
# ============================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_login() {
    log "检查 HuggingFace 登录状态..."
    if ! huggingface-cli whoami; then
        log "❌ 错误: 未登录 HuggingFace"
        log "请运行: huggingface-cli login"
        exit 1
    fi
    log "✅ 已登录 HuggingFace"
}

init_git_lfs() {
    log "初始化 git-lfs..."
    git lfs install 2>/dev/null || true
    log "✅ git-lfs 已初始化"
}

create_repos() {
    log "创建 HuggingFace 仓库（如果不存在）..."
    
    # 创建数据仓库
    huggingface-cli repo create GenHOI-data --type dataset 2>/dev/null || log "数据仓库已存在"
    
    # 创建模型仓库
    huggingface-cli repo create GenHOI --type model 2>/dev/null || log "模型仓库已存在"
    
    log "✅ 仓库检查完成"
}

upload_file() {
    local src="$1"
    local dst="$2"
    local repo="$3"
    local repo_type="$4"
    
    if [ -f "$src" ]; then
        log "上传文件: $src -> $dst"
        huggingface-cli upload "$repo" "$src" "$dst" --repo-type "$repo_type" && \
            log "✅ 完成: $dst" || log "⚠️ 失败: $dst"
    else
        log "⚠️ 文件不存在: $src"
    fi
}

upload_folder() {
    local src="$1"
    local dst="$2"
    local repo="$3"
    local repo_type="$4"
    
    if [ -d "$src" ]; then
        log "上传目录: $src -> $dst"
        huggingface-cli upload "$repo" "$src" "$dst" --repo-type "$repo_type" && \
            log "✅ 完成: $dst" || log "⚠️ 失败: $dst"
    else
        log "⚠️ 目录不存在: $src"
    fi
}

# ============================================
# 主上传流程
# ============================================
main() {
    log "=========================================="
    log "GenHOI HuggingFace 上传脚本"
    log "=========================================="
    log "数据仓库: $HF_DATA_REPO"
    log "模型仓库: $HF_MODEL_REPO"
    log "本地路径: $BASE_DIR"
    log "=========================================="
    
    # 前置检查
    check_login
    init_git_lfs
    create_repos
    
    # ==========================================
    # 1. 上传到数据仓库 (GenHOI-data)
    # ==========================================
    log ""
    log "=========================================="
    log "第一部分: 上传到数据仓库 ($HF_DATA_REPO)"
    log "=========================================="
    
    # 1.1 上传 demo 数据
    log ""
    log "--- [1/4] 上传 demo 数据 (~241MB) ---"
    upload_folder "$BASE_DIR/demo" "demo" "$HF_DATA_REPO" "dataset"
    
    # 1.2 上传 assets
    log ""
    log "--- [2/4] 上传 assets (~5MB) ---"
    upload_folder "$BASE_DIR/assets" "assets" "$HF_DATA_REPO" "dataset"
    
    # 1.3 上传评估模型 (如果存在)
    log ""
    log "--- [3/4] 上传评估模型 ---"
    if [ -f "$BASE_DIR/tools/eval_fvd/i3d_pretrained_400.pt" ]; then
        upload_file "$BASE_DIR/tools/eval_fvd/i3d_pretrained_400.pt" "eval_models/i3d_pretrained_400.pt" "$HF_DATA_REPO" "dataset"
    else
        log "⚠️ i3d_pretrained_400.pt 不存在，跳过"
    fi
    
    if [ -f "$BASE_DIR/tools/eval_fvd/resnet-50-kinetics.pth" ]; then
        upload_file "$BASE_DIR/tools/eval_fvd/resnet-50-kinetics.pth" "eval_models/resnet-50-kinetics.pth" "$HF_DATA_REPO" "dataset"
    else
        log "⚠️ resnet-50-kinetics.pth 不存在，跳过"
    fi
    
    # 1.4 上传 GenHOI_VACE 模型 (如果存在)
    log ""
    log "--- [4/4] 上传 GenHOI_VACE 模型 ---"
    if [ -d "$BASE_DIR/models/GenHOI_VACE" ]; then
        upload_folder "$BASE_DIR/models/GenHOI_VACE" "models/GenHOI_VACE" "$HF_DATA_REPO" "dataset"
    else
        log "⚠️ GenHOI_VACE 目录不存在，跳过"
    fi
    
    # ==========================================
    # 2. 上传到模型仓库 (GenHOI)
    # ==========================================
    log ""
    log "=========================================="
    log "第二部分: 上传到模型仓库 ($HF_MODEL_REPO)"
    log "=========================================="
    
    # 2.1 上传 GenHOI 微调权重
    log ""
    log "--- [1/2] 上传 GenHOI_wan_flf.consolidated (~31GB) ---"
    log "注意: 这是大文件，预计需要数小时"
    if [ -f "$BASE_DIR/models/GenHOI_wan_flf.consolidated" ]; then
        upload_file "$BASE_DIR/models/GenHOI_wan_flf.consolidated" "GenHOI_wan_flf.consolidated" "$HF_MODEL_REPO" "model"
    else
        log "⚠️ GenHOI_wan_flf.consolidated 不存在，跳过"
    fi
    
    # 2.2 上传 Wan-AI 基础模型 (如果需要)
    log ""
    log "--- [2/2] 上传 Wan-AI 基础模型 ---"
    if [ -d "$BASE_DIR/models/Wan-AI" ]; then
        log "发现 Wan-AI 目录，正在上传..."
        upload_folder "$BASE_DIR/models/Wan-AI" "Wan-AI" "$HF_MODEL_REPO" "model"
    else
        log "⚠️ Wan-AI 目录不存在，跳过"
        log "提示: Wan2.1 基础模型可以从官方仓库下载"
    fi
    
    # ==========================================
    # 完成
    # ==========================================
    log ""
    log "=========================================="
    log "🎉 上传任务完成！"
    log "=========================================="
    log ""
    log "数据仓库: https://huggingface.co/datasets/$HF_DATA_REPO"
    log "模型仓库: https://huggingface.co/$HF_MODEL_REPO"
    log ""
    log "请检查仓库并更新 README.md 中的下载链接"
    log "=========================================="
}

# 运行主函数
main "$@"