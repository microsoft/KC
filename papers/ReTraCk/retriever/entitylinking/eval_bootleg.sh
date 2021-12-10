# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

data_name=$1
#exp_name=$2
if [ $data_name == "GrailQA" ]
then
  python retriever/entitylinking/bootleg/bootleg_aml/eval.py --use_gpu --config "${RETRACK_HOME}/retriever/entitylinking/bootleg/bootleg_aml/configs/grailqa_config.json" --pretrain_checkpoint "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/pretrained_bert_models/" --checkpoint "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/GrailQA/model/grailqa_bootleg_wo_prior_reproduce_2345/model4.pt" --base_dir "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/GrailQA/" --experiment_name "grailqa_bootleg_wo_prior_reproduce" --outputs_dir "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/results/" --test_file "dev_bootleg_data_filter_pred_ner.jsonl"
  python retriever/entitylinking/eval/eval_grailqa.py --gold_fn "${DATA_PATH}/Dataset/GrailQA/aligned/dev_v1_aligned.json" --pred_fn "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/results/grailqa_bootleg_wo_prior_reproduce_2345/dev_bootleg_data_filter_pred_ner/eval/model4/topk_predictions.jsonl"
else
  python retriever/entitylinking/bootleg/bootleg_aml/eval.py --use_gpu --config "${RETRACK_HOME}/retriever/entitylinking/bootleg/bootleg_aml/configs/webqsp_distant_config.json" --pretrain_checkpoint "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/pretrained_bert_models/" --checkpoint "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/WebQSP/model/distant_fix_offset_bootleg_woprior_webqsp_train_epoch_50_small_64_dim_relation_50_1234/model30.pt" --base_dir "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/WebQSP/" --experiment_name "webqsp_bootleg_wo_prior_reproduce" --outputs_dir "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/results/" --test_file "webqsp_test_bert_ner_year_v1_no_filter.jsonl"
  python retriever/entitylinking/eval/eval_webqsp.py --pred_fn "${DATA_PATH}/KBSchema/EntityLinking/Bootleg/results/webqsp_bootleg_wo_prior_reproduce_1234/webqsp_test_bert_ner_year_v1_no_filter/eval/model30/topk_predictions.jsonl"
fi