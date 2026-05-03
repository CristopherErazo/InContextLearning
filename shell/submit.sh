#!/bin/bash

experiment_name='fix_data_uniform'
K=8
rank=16
BS=(5000)

# ranks=(2 4 8 16)

for bs in "${BS[@]}"
do
    python -u ./scripts/test.py\
        extra_args.experiment_name=$experiment_name\
        model_args.rank=$rank\
        data_args.K=$K\
        data_args.batch_size=$bs
done
