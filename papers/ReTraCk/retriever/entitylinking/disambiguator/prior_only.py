# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any
import pickle
from retriever.entitylinking.disambiguator.base import EntityDisambiguator
import os
import json


class PriorDisambiguator(EntityDisambiguator):

    def __init__(self, config: Dict[str, Any]):
        super(PriorDisambiguator, self).__init__(config)
        print("loadding {}".format(os.path.join(config['base_data_dir'], config['prior_path'])))
        self.prior = PriorDisambiguator.load_prior_file(os.path.join(config['base_data_dir'], config['prior_path']))

    @staticmethod
    def load_prior_file(fn):
        with open(fn, mode="rb") as fp:
            prior = pickle.load(fp)
            return prior

    def predict(self,
                mentions: Dict[str, Any],
                topk=3) -> Dict[str, Any]:

        toks = mentions["tokens"]
        mentions["topk_entities"] = []
        mentions["topk_scores"] = []
        mentions["boundary"] = []

        for ent in mentions["mentions"]:
            if ent["type"] == "entity":
                mention = " ".join(toks[ent["start"]:ent["end"]])
                if mention in self.prior:
                    tmp_ent_list = [c[0] for c in self.prior[mention]]
                    tmp_score_list = [c[1] for c in self.prior[mention]]
                    mentions["topk_entities"].append(tmp_ent_list)
                    mentions["topk_scores"].append(tmp_score_list)
                    mentions["boundary"].append((ent["start"], ent["end"]))
        return mentions

    def predict_one_file(self,
                         in_fn,
                         out_fn):
        with open(in_fn, encoding='utf-8', mode='r') as fp:
            with open(out_fn, encoding='utf-8', mode='w') as re_fp:
                for line in fp:
                    sent = json.loads(line.strip())
                    tokens = sent["bert_tokens"]
                    sent["prior_baseline"] = []
                    for ent in sent["ner_output"]:
                        if ent[0] == "entity":
                            mention = " ".join(tokens[ent[1]:ent[2]])
                            if mention in self.prior:
                                sent["prior_baseline"].append(["entity", ent[1], ent[2], self.prior[mention][0][0]])
                    re_fp.write("{}\n".format(json.dumps(sent)))
