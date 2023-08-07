# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEEDS=(171 354 550 667 985)
TASKS=(business sci sports world)
SEED=$1
GPU_ID=$2
task=$3

python main_ood.py \
    --output_dir ./results/ag/${task}/CLM_5 \
    --method CLM \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix ${task} \
    --train_file OOD_dataset/ag/train/${task}.txt \
    --validation_file OOD_dataset/ag/test/${task}.txt \
    --test_file OOD_dataset/ag/ood/${task}.txt
