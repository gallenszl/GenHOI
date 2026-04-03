python examples/wanvideo/swap_infer.py \
    --output_dir results/swap_vace_cross_me \
    --data_csv /root/paddlejob/workspace/shenzhelun/data/gongyingshang_cross_our_paths_bbox_replaced.csv \
    --gpus 0,1,2,3,4,5,6,7 \
    --max_num_frames 81 \
    --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
    --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors


# python examples/wanvideo/selfswap_infer.py \
#     --output_dir results/selfswap_vace_ff_debug \
#     --data_csv demo/demo_selfswap.csv \
#     --gpus 1 \
#     --max_num_frames 81 \
#     --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
#     --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors

# python examples/wanvideo/selfswap_infer.py \
#     --output_dir results/selfswap_demo \
#     --data_csv data/AnchorCrafter-400_405f/dataset_select_f50.csv \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --max_num_frames 81 \
#     --model_path models/GenHOI_VACE/step-5000-gate_attn.safetensors \
#     --lora_path models/GenHOI_VACE/step-1700-lora-gate-720.safetensors