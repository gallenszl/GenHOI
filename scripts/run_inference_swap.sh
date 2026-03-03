python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_vace_ff \
    --data_csv data/long_video_swap/swap.csv \
    --gpus 0,1,2,3,4,5,6,7 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors