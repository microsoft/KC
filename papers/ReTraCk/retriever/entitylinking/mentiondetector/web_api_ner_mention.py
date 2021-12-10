# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any, List
import requests
import json
from retriever.entitylinking.mentiondetector.base import MentionDetector


class WebAPINERMentionDetector(MentionDetector):

    def __init__(self, config: Dict[str, Any]):

        super(WebAPINERMentionDetector, self).__init__(config)

        # @TODO Move host to config
        if config['world'] == 'GrailQA':
            self.ner_uri = f"http://localhost:{config['grailqa_ner_port']}/kbqa/api/v1.0/ner"
        elif config['world'] == 'WebQSP':
            self.ner_uri = f"http://localhost:{config['webqsp_ner_port']}/kbqa/api/v1.0/ner"
        else:
            raise ValueError(f"Unknown world: {config['world']}")

        response = requests.post(self.ner_uri, json={"sentence": ""})
        if response.status_code != 201:
            raise EnvironmentError(f"NER URL not up: {self.ner_uri}")

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
        data = {
            "sentence": sentence
        }

        output = requests.post(self.ner_uri, json=data)

        ner_output = json.loads(output.text)
        tokens = ner_output["bert_tokens"]
        mentions = []
        for x in ner_output["ner_output"]:
            mentions.append({"start": x[1],
                             "end": x[2],
                             "type": x[0]})
        return {"tokens": tokens,
                "mentions": mentions}


if __name__ == "__main__":
    conf = {
        "grailqa_ner_port": 8009,
        "webqsp_ner_port": 8010,
        "world": "GrailQA"
    }

    mention_detector = WebAPINERMentionDetector(conf)
    print(mention_detector.predict("welcome to beijing ."))
