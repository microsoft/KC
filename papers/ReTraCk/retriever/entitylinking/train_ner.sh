# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

data_name=$1
cd ${RETRACK_HOME}/retriever/entitylinking/
cp train_ner.py BERT-NER/
cd BERT-NER/
if [ $data_name == "GrailQA" ]
then
  CUDA_VISIBLE_DEVICES=0 python train_ner.py --data_dir "${DATA_PATH}/KBSchema/EntityLinking/NER/GrailQA/train/" --bert_model "bert-base-uncased" --task_name "grailqa" --output_dir "${DATA_PATH}/KBSchema/EntityLinking/NER/GrailQA/model/" --do_train --do_eval --do_lower_case --num_train_epochs 5
else
  CUDA_VISIBLE_DEVICES=0 python train_ner.py --data_dir "${DATA_PATH}/KBSchema/EntityLinking/NER/WebQSP/train/" --bert_model "bert-base-uncased" --task_name "webqsp" --output_dir "${DATA_PATH}/KBSchema/EntityLinking/NER/WebQSP/model/" --do_train --do_eval --do_lower_case --num_train_epochs 5
fi