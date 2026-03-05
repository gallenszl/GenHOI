# python examples/wanvideo/swap_infer.py \
#     --output_dir results/swap_vace_ff_debug \
#     --data_csv data/long_video_swap/swap.csv \
#     --gpus 0 \
#     --max_num_frames 81 \
#     --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
#     --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors


# python examples/wanvideo/selfswap_infer.py \
#     --output_dir results/selfswap_vace_ff_debug \
#     --data_csv demo/demo_selfswap.csv \
#     --gpus 1 \
#     --max_num_frames 81 \
#     --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
#     --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors

python examples/wanvideo/test_selfswap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/selfswap_81 \
    --data_csv data/AnchorCrafter-400_405f/dataset_select_f50.csv \
    --gpus 2 \
    --max_num_frames 81 \
    --is_fl