# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from retriever.entitylinking.disambiguator.prior_only import PriorDisambiguator
import argparse
from retriever.configs import config_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_fn", type=str, default="/home/v-shuach/Workspace/ReTraCkData/KBSchema/EntityLinking/NER/GrailQA/inference/dev_v1_aligned_bert_ner_datetime.json")
    parser.add_argument("--out_fn", type=str, default="/home/v-shuach/Workspace/ReTraCkData/KBSchema/EntityLinking/Prior/GrailQA/dev_v1_aligned_bert_ner_datetime_topk.json")
    parser.add_argument("--config_path", type=str, default="/home/v-shuach/Workspace/ListQACode/configs/shuang_retriever_config.json")
    args = parser.parse_args()
    print('Using config: ' + args.config_path)
    config = config_utils.get_config(args.config_path)
    print(config)
    prior_baseline = PriorDisambiguator(config)
    prior_baseline.predict_one_file(args.in_fn, args.out_fn)
