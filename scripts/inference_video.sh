# python examples/wanvideo/test_swap.py \
#     --model_path models/GenHOI_wan_flf.consolidated \
#     --output_dir results/swap_81 \
#     --data_csv data/long_video_swap/swap.csv \
#     --gpus 2 \
#     --max_num_frames 81

python examples/wanvideo/test_selfswap.py \
    --model_path models/GenHOI_wan_flf.consolidated \
    --output_dir results/selfswap_81 \
    --data_csv data/AnchorCrafter-400_405f/dataset_select_f50.csv \
    --gpus 2 \
    --max_num_frames 81 \
    --is_fl