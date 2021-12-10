# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

data_name=$1
cd ${RETRACK_HOME}/retriever/entitylinking/
cp inf_ner.py BERT-NER/
cd BERT-NER/
if [ $data_name == "GrailQA" ]
then
  python inf_ner.py --checkpoint "$DATA_PATH/KBSchema/EntityLinking/NERModel/GrailQA" --in_fn "${DATA_PATH}/Dataset/GrailQA/aligned/dev_v1_aligned.json" --out_fn "$DATA_PATH/KBSchema/EntityLinking/NER/GrailQA/results/dev_v1_aligned_ner.json" --dataset GrailQA
else
  python inf_ner.py --checkpoint "$DATA_PATH/KBSchema/EntityLinking/NERModel/WebQSP" --in_fn "${DATA_PATH}/Dataset/WebQSP/aligned/webqsp_test_aligned.jsonl" --out_fn "$DATA_PATH/KBSchema/EntityLinking/NER/WebQSP/results/webqsp_test_aligned_ner.json" --dataset WebQSP
fi