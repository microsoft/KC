# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/ROSTD/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --kd_teacher_dir ./results/ROSTD/CLM_5 \
    --intermediate_mode middle \
    --use_mse_loss both \
    --log_suffix ROSTD \
    --train_file ./OOD_dataset/ROSTD/ROSTD_train.txt \
    --validation_file ./OOD_dataset/ROSTD/ROSTD_test.txt \
    --test_file ./OOD_dataset/ROSTD/ROSTD_ood.txt

