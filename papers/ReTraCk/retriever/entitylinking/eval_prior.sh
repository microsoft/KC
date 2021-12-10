# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

data_name=$1
#exp_name=$2
if [ $data_name == "GrailQA" ]
then
  python retriever/entitylinking/disambiguator/eval_prior.py --in_fn "$DATA_PATH/KBSchema/EntityLinking/NER/GrailQA/results/dev_v1_aligned_ner.json" --out_fn "${DATA_PATH}/KBSchema/EntityLinking/Prior/GrailQA/dev_v1_aligned_ner_prior.json" --config_path "${RETRACK_HOME}/configs/shuang_retriever_config.json"
  python retriever/entitylinking/eval/eval_grailqa.py --gold_fn "${DATA_PATH}/Dataset/GrailQA/aligned/dev_v1_aligned.json" --pred_fn "${DATA_PATH}/KBSchema/EntityLinking/Prior/GrailQA/dev_v1_aligned_ner_prior.json" --model "prior"
else
  python retriever/entitylinking/disambiguator/eval_prior.py --in_fn "$DATA_PATH/KBSchema/EntityLinking/NER/WebQSP/results/webqsp_test_aligned_ner.json" --out_fn "${DATA_PATH}/KBSchema/EntityLinking/Prior/WebQSP/webqsp_test_aligned_ner_prior.json" --config_path "${RETRACK_HOME}/configs/shuang_retriever_config.json"
  python retriever/entitylinking/eval/eval_webqsp.py --pred_fn "${DATA_PATH}/KBSchema/EntityLinking/Prior/WebQSP/webqsp_test_aligned_ner_prior.json" --model "prior"
fi
