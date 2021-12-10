# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any, List
from retriever.entitylinking.mentiondetector.base import MentionDetector
from retriever.entitylinking.bert_ner.bert import NER
import os
from retriever.configs import config_utils
import argparse

class NERMentionDetector(MentionDetector):

    def __init__(self, config: Dict[str, Any]):
        super(NERMentionDetector, self).__init__(config)
        if config['world'] == "GrailQA":
            self.ner = NER(os.path.join(config["base_data_dir"], config["grailqa_ner_model_path"]))
        elif config['world'] == "WebQSP":
            self.ner = NER(os.path.join(config["base_data_dir"], config["webqsp_ner_model_path"]))
        else:
            raise ValueError(f"unknown world: {config['world']}")

    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Input a sentence, return a dictionary including two fields:

            tokens: e.g., ["hello", "world]

            mentions: e.g., [{"start": 0, "end": 1, "type"}]

        Parameters
        ----------
        sentence: str

        Returns: Dict[str, Any]
        -------

        """
        ner_output = self.ner.run_ner(sentence)
        tokens = ner_output["bert_tokens"]
        mentions = []
        for x in ner_output["ner_output"]:
            mentions.append({
                "start": x[1],
                "end": x[2],
                "type": x[0]
            })
        return {
            "tokens": tokens,
            "mentions": mentions
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="retriever/configs/config.json")
    args = parser.parse_args()

    config = config_utils.get_config(args.config_path)

    mention_detector = NERMentionDetector(config)
    print(mention_detector.predict("welcome to beijing ."))
