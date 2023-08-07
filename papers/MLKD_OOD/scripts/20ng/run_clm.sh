# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEEDS=(171 354 550 667 985)
TASKS=(comp misc pol rec rel sci)
SEED=$1
GPU_ID=$2
task=$3

python main_ood.py \
    --output_dir ./results/20ng/${task}/CLM_5 \
    --method CLM \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix ${task} \
    --num_train_epochs 30 \
    --train_file OOD_dataset/20ng/train/${task}.txt \
    --validation_file OOD_dataset/20ng/test/${task}.txt \
    --test_file OOD_dataset/20ng/ood/${task}.txt
