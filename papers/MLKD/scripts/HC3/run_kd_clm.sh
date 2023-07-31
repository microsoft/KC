# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/HC3/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix hc3 \
    --kd_teacher_dir ./results/HC3/CLM_5 \
    --intermediate_mode middle \
    --train_from_scratch \
    --train_file ./OOD_dataset/HC3/hc3_train.txt \
    --validation_file ./OOD_dataset/HC3/hc3_test.txt \
    --test_file ./OOD_dataset/HC3/hc3_ood.txt
