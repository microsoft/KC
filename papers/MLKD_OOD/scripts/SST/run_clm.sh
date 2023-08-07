# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEEDS=(171 354 550 667 985)

SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/SST/CLM_5 \
    --method CLM \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix sst \
    --train_file ./OOD_dataset/SST/sst_train.txt \
    --validation_file ./OOD_dataset/SST/sst_test.txt \
    --test_file ./OOD_dataset/SST/sst_oos_test.txt
