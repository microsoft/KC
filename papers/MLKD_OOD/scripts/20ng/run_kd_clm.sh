# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2
task=$3

python main_ood.py \
    --output_dir ./results/20ng/${task}/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --kd_teacher_dir ./results/20ng/${task}/CLM_5 \
    --intermediate_mode middle \
    --num_train_epochs 30 \
    --log_suffix ${task} \
    --train_file OOD_dataset/20ng/train/${task}.txt \
    --validation_file OOD_dataset/20ng/test/${task}.txt \
    --test_file OOD_dataset/20ng/ood/${task}.txt
