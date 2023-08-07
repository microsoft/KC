# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix clinc150 \
    --kd_teacher_dir ./results/CLM_5 \
    --intermediate_mode middle \
    --train_file ./OOD_dataset/CLINIC150/clinc150_train.txt \
    --validation_file ./OOD_dataset/CLINIC150/clinc150_test.txt \
    --test_file ./OOD_dataset/CLINIC150/clinc150_oos_test.txt
