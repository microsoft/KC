# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/SST/KD_scratch_CLM_5_intermediate_kl \
    --method KD_CLM \
    --train_from_scratch \
    --loop_type linear_tea_mse \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --kd_teacher_dir ./results/SST/CLM_5 \
    --intermediate_mode middle \
    --temperature 0.5 \
    --log_suffix sst \
    --train_file ./OOD_dataset/SST/sst_train.txt \
    --validation_file ./OOD_dataset/SST/sst_test.txt \
    --test_file ./OOD_dataset/SST/sst_oos_test_${SEED}.txt

