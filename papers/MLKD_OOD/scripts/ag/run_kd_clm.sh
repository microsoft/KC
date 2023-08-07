# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2
task=$3

python main_ood.py \
    --output_dir ./results/ag/${task}/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --kd_teacher_dir ./results/ag/${task}/CLM_5 \
    --intermediate_mode middle \
    --log_suffix ${task} \
    --train_file OOD_dataset/ag/train/${task}.txt \
    --validation_file OOD_dataset/ag/test/${task}.txt \
    --test_file OOD_dataset/ag/ood/${task}.txt
