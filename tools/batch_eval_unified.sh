#!/bin/bash
# ==============================================================================
# GenHOI Unified Evaluation Script
# ==============================================================================
# Description: Simplified evaluation script with unified output
# All results are saved to a single JSON file without intermediate directories
#
# Usage:
#   bash tools/batch_eval_unified.sh <base_dir> [sample_duration] [device]
#
# Examples:
#   bash tools/batch_eval_unified.sh /path/to/results 81 cuda
#   bash tools/batch_eval_unified.sh /path/to/results 401 cpu
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================
base_dir=${1:-""}
sample_duration=${2:-81}
device=${3:-cuda}

# Number of samples to evaluate (set to 0 to evaluate all)
MAX_SAMPLES=${MAX_SAMPLES:-0}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Output file
OUTPUT_FILE="${base_dir}/all_metrics.json"

# ==============================================================================
# Validation
# ==============================================================================
if [ -z "$base_dir" ]; then
    echo "Error: base_dir is required"
    echo ""
    echo "Usage: bash tools/batch_eval_unified.sh <base_dir> [duration] [device]"
    echo ""
    echo "Example:"
    echo "  bash tools/batch_eval_unified.sh /path/to/results 81 cuda"
    exit 1
fi

if [ ! -d "$base_dir" ]; then
    echo "Error: Directory does not exist: $base_dir"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         GenHOI Unified Evaluation (Simplified)                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Base directory:    $base_dir"
echo "  Sample duration:   $sample_duration frames"
echo "  Device:            $device"
echo "  Max samples:       $MAX_SAMPLES (0 = all)"
echo "  Output file:       $OUTPUT_FILE"
echo ""

# Activate conda environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate wan_new

# ==============================================================================
# File Organization (in base_dir, no subdirectories)
# ==============================================================================
echo "Organizing video files..."

copy_and_rename() {
    local pattern=$1
    local suffix=$2
    local count=0
    
    for src in $(ls "$base_dir"/sample_*_allclips/${pattern} 2>/dev/null | head -n ${MAX_SAMPLES:-999999}); do
        if [ -f "$src" ]; then
            base=$(basename "$(dirname "$src")")
            prefix=${base%_allclips}
            dest="$base_dir/${prefix}_${suffix}.mp4"
            cp "$src" "$dest"
            count=$((count + 1))
        fi
    done
    
    echo "  ✓ Copied $count files: ${pattern} -> *_${suffix}.mp4"
}

copy_and_rename "all_generated.mp4" "generated" "$base_dir"
copy_and_rename "all_gt.mp4" "gt" "$base_dir"
copy_and_rename "all_handpose.mp4" "handpose" "$base_dir"
copy_and_rename "all_ref.mp4" "ref" "$base_dir"

echo ""

# ==============================================================================
# Metric 1: FVD + FID-VID (using DisCo metric_center.py)
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Step 1/3: Computing FVD + FID-VID                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Create separate directories for gen and gt videos (required by metric_center.py)
gen_dir="${base_dir}_gen"
gt_dir="${base_dir}_gt"
mkdir -p "$gen_dir" "$gt_dir"

echo "  Creating gen/gt directories..."
cp -f "$base_dir"/*_generated.mp4 "$gen_dir/" 2>/dev/null || true
cp -f "$base_dir"/*_gt.mp4 "$gt_dir/" 2>/dev/null || true

gen_count=$(ls "$gen_dir"/*.mp4 2>/dev/null | wc -l)
gt_count=$(ls "$gt_dir"/*.mp4 2>/dev/null | wc -l)
echo "  ✓ Copied $gen_count generated videos to $gen_dir"
echo "  ✓ Copied $gt_count GT videos to $gt_dir"

if [ "$gen_count" -eq 0 ] || [ "$gt_count" -eq 0 ]; then
    echo "  ⚠ No generated or GT videos found, skipping FVD"
else
    # Calculate relative paths for metric_center.py
    base_parent_dir=$(dirname "$base_dir")
    relative_gen_dir=$(realpath --relative-to="$base_parent_dir" "$gen_dir")"/"
    relative_gt_dir=$(realpath --relative-to="$base_parent_dir" "$gt_dir")"/"
    
    echo "  Base parent dir: $base_parent_dir"
    echo "  Relative gen dir: $relative_gen_dir"
    echo "  Relative gt dir: $relative_gt_dir"
    echo ""
    
    # Use local FVD module (extracted from DisCo) for FVD calculation
    cd "$PROJECT_ROOT" || exit
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/eval_fvd/fvd.py" \
        --gen_dir "$gen_dir" \
        --gt_dir "$gt_dir" \
        --sample_duration $sample_duration \
        --weights_dir "$SCRIPT_DIR/eval_fvd" \
        --device "$device" \
        --output "$base_dir/metrics_video.json" \
        --modes FVD-3DRN50 FVD-3DInception

    echo ""
    echo "  ✓ FVD + FID-VID completed"
    echo "  ✓ Results saved to: $base_dir/metrics_video.json"
fi

echo ""

# ==============================================================================
# Metric 2: FID + PSNR + SSIM
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Step 2/3: Computing FID, PSNR, SSIM                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_ROOT" || exit
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/fid.py" \
    --video_dir "$base_dir/" \
    --device "$device" \
    --output "$base_dir/metrics_frame.csv"

echo ""
echo "  ✓ FID, PSNR, SSIM evaluation completed"
echo "  ✓ Results saved to: $base_dir/metrics_frame.csv"
echo ""

# ==============================================================================
# Metric 3: Object-CLIP (OC) - Optional
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Step 3/3: Computing Object-CLIP (OC) Scores                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$SCRIPT_DIR/oc_metric_with_viz.py" ]; then
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/batch_oc_eval.py" \
        --root "$base_dir" \
        --stride 1 \
        --aggregate mean \
        --device "$device"

    echo ""
    echo "  ✓ OC evaluation completed"
    echo "  ✓ Results saved to: $base_dir/oc_scores.csv"
else
    echo "  ⚠ oc_metric_with_viz.py not found, skipping OC evaluation"
fi

echo ""

# ==============================================================================
# Merge All Results
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Merging All Results                                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python "$SCRIPT_DIR/merge_results_unified.py" \
    --base_dir "$base_dir" \
    --output "$OUTPUT_FILE"

echo ""
echo "  ✓ Results merged to: $OUTPUT_FILE"
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Evaluation Complete                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "All results saved to: $OUTPUT_FILE"
echo ""
echo "View results:"
echo "  cat $OUTPUT_FILE"
echo ""
echo "✨ Evaluation completed successfully!"
echo ""