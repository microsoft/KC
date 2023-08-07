# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEEDS=(171 354 550 667 985)

SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/ROSTD/CLM_5 \
    --method CLM \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix ROSTD \
    --train_file ./OOD_dataset/ROSTD/ROSTD_train.txt \
    --validation_file ./OOD_dataset/ROSTD/ROSTD_test.txt \
    --test_file ./OOD_dataset/ROSTD/ROSTD_ood.txt
