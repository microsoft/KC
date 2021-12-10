# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
from typing import List, Tuple
from pytorch_transformers.tokenization_bert import BertTokenizer
from retriever.schema_retriever.utils import json_helper, dict_helper
import retriever.schema_retriever.constants_schema_retriever as constants
import retriever.schema_retriever.dense_retriever.blink as blink
from retriever.schema_retriever.dense_retriever.blink import main_dense
from retriever.schema_retriever.dense_retriever.blink.candidate_ranking import utils
import retriever.schema_retriever.dense_retriever.blink.biencoder.train_biencoder as train
import retriever.schema_retriever.dense_retriever.blink.biencoder.get_candidate_emb as get_cand
from retriever.configs import config_utils


class DenseSchemaRetriever(object):

    def __init__(self, config=None):

        # Load default config
        self.config = json_helper.from_str(constants.defaultConfig)

        if not isinstance(config, dict) and not None:
            config = config_utils.get_config(config)

        # Merge config, is necessary
        if config:
            self.config = dict_helper.merge_two_dicts(self.config, config)

        print("---------------------")
        print("Used DenseSchemaRetriever configuration:\n", self.config)
        print("---------------------")

        # Load models
        self.models = self.get_demo_models()

        # json_helper.to_file(self.config, constants.defaultConfigFileName, constants.defaultConfigIndents)

    def get_model_config(self, world, schema_type):

        model_config = config_utils.process_config_paths(self.config["model"][world][schema_type])
        return argparse.Namespace(**model_config)

    @staticmethod
    def load_model(args, logger=None):  # world=GrailQA|WebQSP, type=Relation|Class
        return main_dense.load_models(args, logger=None)

    @staticmethod
    def model_run(args, model, data):
        _, _, _, _, _, predictions, scores, = main_dense.run(args, utils.get_logger(), *model, test_data=data)
        return predictions, scores

    def train(self):
        config = self.config["train"]
        train.main(config)

    # Generate embedding for the decoding vocabulary
    def generate_emb(self, world, pred_type):
        config = self.config["generate_emb"][world][pred_type]
        get_cand.main(config)

    def get_demo_models(self):

        print(">> Loading demo models...")

        models = [('WebQSP', "Relation"), ('WebQSP', "Class"), ('GrailQA', 'Relation'), ('GrailQA', 'Class')]

        res = {}
        for m in models:
            model_args = self.get_model_config(m[0], m[1])
            res[m] = blink.main_dense.load_models(model_args, logger=None)
        return res

    def batch_predict(self, world, input_file=None, output_path=None):

        if input_file is None:
            input_file = self.config["predict"][world]["input_file"]

        #if output_path is None:
        #    output_path = self.config["predict"][pred_type]["res_path"]

        res_cls = self.batch_predict_per_type(world, "Class", input_file, output_path)

        res_rel = self.batch_predict_per_type(world, "Relation", input_file, output_path)

        return res_cls, res_rel

    def batch_predict_per_type(self, world, pred_type, input_file=None, output_path=None):

        # Check that both input and output locations exist

        if input_file is None:
            input_file = self.config["predict"][world]["input_file"]

        if output_path is None:
            output_file = os.path.join(self.config["predict"][world]["res_path"], f'{pred_type}.res')

        data_f = open(input_file, "r")
        data = json.load(data_f)

        config = self.config["predict"][world][pred_type]
        args = argparse.Namespace(**config)

        model = self.load_model(args)
        # model = self.models[(world, pred_type)]

        predictions, scores = self.model_run(args, model, data)
        pred_normalization = json_helper.from_file(self.config["predict"][pred_type])

        res = []
        for i in range(len(predictions)):
            exp = {
                "predictions": "\t".join(str(pred_normalization[j]) for j in list(predictions[i])),
                "scores": '\t'.join(str(i) for i in list(scores[i])),
            }
            res.append(exp)

        json_helper.to_file(res, output_file, 4)

        return res

    def merge_results(self, class_pred_file, rel_pred_file, dataset, mode):

        out_dir = self.config["predict"][dataset]["res_path"]

        cnt = 0

        relation_pres = []
        class_pres = []
        k_rel = 150
        k_cls = 100

        class_preds = json.load(open(class_pred_file, 'r', encoding='utf8'))
        for pred in class_preds:
            curr_pre = pred["predictions"].split('\t')
            curr_pre_score = pred["scores"].split('\t')
            tmp = []
            for n in range(k_rel):
                tmp.append((curr_pre[n], curr_pre_score[n]))
            class_pres.append(tmp)

        rel_preds = json.load(open(rel_pred_file, 'r', encoding='utf8'))
        for pred in rel_preds:
            curr_pre = pred["predictions"].split('\t')
            curr_pre_score = pred["scores"].split('\t')
            tmp = []
            for n in range(k_cls):
                tmp.append((curr_pre[n], curr_pre_score[n]))
            relation_pres.append(tmp)

        output_file = f'{out_dir}/{dataset}_{mode}_res_Rel{k_rel}_Cls{k_cls}.jsonl'
        ofp = open(output_file, 'w', encoding='utf8')

        for item in class_pres:
            exp = {
                "qid": cnt,
                "classes": class_pres[cnt],
                "relations": relation_pres[cnt],
            }

            ofp.write(json.dumps(exp) + "\n")

            cnt += 1

        ofp.close()

        return output_file, []

    def dense_api(self, question: str, world: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:

        if not self.models:
            raise Exception("Models should be loaded before calling dense_api!")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        question = [{
            "id": 0,
            "label": "",
            "label_id": -1,
            "context_left": "".lower(),
            "mention": " ".join(tokenizer.tokenize(question)),
            "context_right": "".lower(),
        }]

        if world not in self.config["model"]:
            print(f'>>> Models for {world} not loaded')
            return [], []

        args_rel = argparse.Namespace(**self.config["model"][world]["Relation"])
        model_rel = self.models[(world, "Relation")]
        predictions_rel, scores_rel = self.model_run(args_rel, model_rel, question)

        args_cls = argparse.Namespace(**self.config["model"][world]["Class"])
        model_cls = self.models[(world, "Class")]
        predictions_cls, scores_cls = self.model_run(args_cls, model_cls, question)

        rel_normalization = json_helper.from_file(self.config["model"]["normalized_relation"])
        cls_normalization = json_helper.from_file(self.config["model"]["normalized_class"])

        rel_res = []
        cls_res = []
        predictions_rel = predictions_rel[0]
        predictions_cls = predictions_cls[0]

        for i in range(len(predictions_rel)):
            rel_res.append((rel_normalization[predictions_rel[i]], float(scores_rel[0][i])))

        for j in range(len(predictions_cls)):
            cls_res.append((cls_normalization[predictions_cls[j]], float(scores_cls[0][j])))

        return cls_res, rel_res

    def evaluate(self, pred_path, gold_path):  # two list of items pred_path=[item,score]

        ks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200]
        recall_rel = {}
        recall_cls = {}
        for k in ks:
            recall_cls[str(k)] = 0
            recall_rel[str(k)] = 0

        pred = json_helper.from_jsonl(pred_path)
        gold = json_helper.from_file(gold_path)
        num = len(pred)

        for id in range(num):
            curr_rel = [i[0] for i in pred[id]["relations"]]
            curr_cls = [j[0] for j in pred[id]["classes"]]
            gold_rel = set(gold[id]["relations"])
            gold_cls = set(gold[id]["classes"])

            for k in ks:
                if len(set(gold_rel)) == 0:
                    recall_rel[str(k)] += 1 / num
                else:
                    recall_rel[str(k)] += (float(len(gold_rel & set(curr_rel[:k]))) / len(gold_rel)) / num
                if len(set(gold_cls)) == 0:
                    recall_cls[str(k)] += 1 / num
                else:
                    recall_cls[str(k)] += (float(len(gold_cls & set(curr_cls[:k]))) / len(gold_cls)) / num
            
        return recall_rel, recall_cls


if __name__ == "__main__":
    sr = DenseSchemaRetriever()
