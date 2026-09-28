#!/bin/bash

experiment_name='inspection_prev'
batch_size=512
total_steps=20000
n_prints_model=20
track_artifacts=True
alpha_lr=3500
matmul_precision="high"
log_to_terminal=True
test_size=2048
seed=1
mask1="prev"
gen_device="cpu"
stop_at_accuracy=0.9

#configurations to run in (d,L) tuples

configurations=(
    "256 128"
    "256 512"
    "1024 128"
    "1024 512"
)

for config in "${configurations[@]}"
do
    read -r d L <<< "$config"
    echo "Running experiment with d $d and L $L. Initiated at $(date)"
    uv run python -u ./scripts/train.py\
        model_args.d_model=$d\
        model_args.seq_len=$L\
        extra_args.experiment_name=$experiment_name\
        data_args.batch_size=$batch_size\
        extra_args.total_steps=$total_steps\
        extra_args.n_prints_model=$n_prints_model\
        extra_args.track_artifacts=$track_artifacts\
        optim_args.alpha_lr=$alpha_lr\
        extra_args.matmul_precision=$matmul_precision\
        extra_args.log_to_terminal=$log_to_terminal\
        data_args.test_size=$test_size\
        extra_args.seed=$seed\
        model_args.mask1=$mask1\
        data_args.gen_device=$gen_device\
        extra_args.stop_at_accuracy=$stop_at_accuracy
    echo "Finished experiment with d $d and L $L. Finished at $(date)"
done

