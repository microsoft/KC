# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import os
from importlib import reload
from bootleg import run
from bootleg.utils.parser_utils import get_full_config
import random
import argparse
import json
random.seed(1234)
reload(logging)
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_top_k_entities(in_fn, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in fp:
                sent = json.loads(line.strip())
                sent["topk_entities"] = []
                sent["topk_scores"] = []
                for j, cand in enumerate(sent["cands"]):
                    candid_info = []
                    for i in range(len(cand)):
                        candid_info.append({"ent": cand[i],
                                            "score": sent["cand_probs"][j][i]})
                    sorted_candid_info = sorted(candid_info, key=lambda x:x["score"], reverse=True)
                    sent["topk_entities"].append([x['ent'] for x in sorted_candid_info if x["ent"] != "-1"])
                    sent["topk_scores"].append([x['score'] for x in sorted_candid_info if x["ent"] != "-1"])
                re_fp.write("{}\n".format(json.dumps(sent)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true', help="whether to use gpu")
    parser.add_argument("--config", type=str, default="grailqa_config.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrain_checkpoint", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument("--use_prior", action='store_true', help="whether to use prior")
    parser.add_argument("--test_file", type=str, default="dev_bootleg_data_filter_pred_ner.jsonl")
    args = parser.parse_args()

    # setup configs
    use_cpu = args.use_gpu
    use_prior = args.use_prior
    # root_dir = args.root_dir
    config_path = args.config
    config_args = get_full_config(config_path)
    print(config_args)
    config_args.run_config.init_checkpoint = args.checkpoint
    config_args.run_config.experiment_name = args.experiment_name
    # set the path for the entity db and candidate map
    config_args.data_config.entity_dir = f'{args.base_dir}/{config_args.data_config.entity_dir}'
    config_args.data_config.data_dir = f'{args.base_dir}/{config_args.data_config.data_dir}'
    config_args.data_config.test_dataset.file = args.test_file
    config_args.data_config.emb_dir = f'{args.base_dir}/{config_args.data_config.emb_dir}'
    config_args.data_config.word_embedding.cache_dir = args.pretrain_checkpoint

    # set the save directory
    config_args.run_config.save_dir = args.outputs_dir

    # set whether to run inference on the CPU
    config_args.run_config.cpu = use_cpu
    config_args.model_config.use_prior = use_prior

    pred_file, _ = run.model_eval(args=config_args, mode="dump_preds", logger=logger, is_writer=True)
    print(pred_file)

    top_k_pred_file = os.path.join(os.path.dirname(pred_file), "topk_predictions.jsonl")
    get_top_k_entities(pred_file, top_k_pred_file)
