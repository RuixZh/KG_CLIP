CUDA_LAUNCH_BLOCKING=1
now=$(date +"%Y%m%d_%H%M%S")

# torchrun \
# 	--nnodes=1 \
# 	--nproc_per_node=4 \


########################################################################

python3 train.py --isTraining \
        --sim_header "Transf" \
        --network_arch "ViT-B/16" \
        --batch_size 15 \
        --max_frames 16 \
        --num_workers 2 \
        --lam_coef 1.0 \
        --eval_freq 1 \
        --va_path "../Action_Reasoning/dataset/v-a-trn82.pkl" \
        --vb_path "../Action_Reasoning/dataset/v-b.pkl" \
        --ab_path "../Action_Reasoning/dataset/a-b.pkl" \
        --kg_dict "../Action_Reasoning/dataset/kg_idx.json" \
        --log_time $now

python3 train.py --isTraining \
        --sim_header "Transf" \
        --network_arch "ViT-B/16" \
        --batch_size 15 \
        --max_frames 16 \
        --num_workers 2 \
        --lam_coef 1.5 \
        --eval_freq 1 \
        --va_path "../Action_Reasoning/dataset/v-a-trn82.pkl" \
        --vb_path "../Action_Reasoning/dataset/v-b.pkl" \
        --ab_path "../Action_Reasoning/dataset/a-b.pkl" \
        --kg_dict "../Action_Reasoning/dataset/kg_idx.json" \
        --log_time $now

python3 train.py --isTraining \
        --sim_header "Transf" \
        --network_arch "ViT-B/16" \
        --batch_size 15 \
        --max_frames 16 \
        --num_workers 2 \
        --lam_coef 0.5 \
        --eval_freq 1 \
        --va_path "../Action_Reasoning/dataset/v-a-trn82.pkl" \
        --vb_path "../Action_Reasoning/dataset/v-b.pkl" \
        --ab_path "../Action_Reasoning/dataset/a-b.pkl" \
        --kg_dict "../Action_Reasoning/dataset/kg_idx.json" \
        --log_time $now
