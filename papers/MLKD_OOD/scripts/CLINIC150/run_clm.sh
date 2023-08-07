# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
SEEDS=(171 354 550 667 985)

SEED=$1
GPU_ID=$2

python main_ood.py \
    --output_dir ./results/CLM_5 \
    --method CLM \
    --seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --log_suffix clinc150 \
    --train_file ./OOD_dataset/CLINIC150/clinc150_train.txt \
    --validation_file ./OOD_dataset/CLINIC150/clinc150_test.txt \
    --test_file ./OOD_dataset/CLINIC150/clinc150_oos_test.txt
