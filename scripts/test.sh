CUDA_LAUNCH_BLOCKING=1
now=$(date +"%Y%m%d_%H%M%S")

# torchrun \
# 	--nnodes=1 \
# 	--nproc_per_node=1 \
python3	train.py \
    --sim_header "Transf" \
    --network_arch "ViT-B/32" \
    --batch_size 128 \
    --max_frames 8 \
    --lam_coef 0.1 \
    --eval_freq 1 \
    --va_path "../Action_Reasoning/dataset/case.pkl" \
    --kg_dict "../Action_Reasoning/dataset/kg_idx.json" \
    --log_time $now
