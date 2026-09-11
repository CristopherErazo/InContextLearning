#!/bin/bash

experiment_name='swap_experiments'
valuesd=(32 64 128 256 512)
valuesV=(512)
# valuesL=(32 64 256 512) # to not repeat the default value of 128, which is already in the first loop
Nit=3

for i in $(seq 1 $Nit)
do
    # add 10 to the seed to avoid overlap with other experiments
    seed=$((i + 180))
    for V in "${valuesV[@]}"
    do
        echo "Running experiment with d $d and seed $seed"
        python -u ./scripts/launcher.py\
            model_args.vocab_size=$V\
            extra_args.experiment_name=$experiment_name\
            extra_args.seed=$seed
    done

    # for L in "${valuesL[@]}"
    # do
    #     echo "Running experiment with sequence length $L and seed $i"
    #     python -u ./scripts/launcher.py\
    #         extra_args.experiment_name=$experiment_name\
    #         model_args.seq_len=$L\
    #         extra_args.seed=$seed
    # done
done


