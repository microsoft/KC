# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Any
# from retriever.entitylinking.mentiondetector.ner_mention import NERMentionDetector
from retriever.entitylinking.mentiondetector.web_api_ner_mention import WebAPINERMentionDetector
from retriever.entitylinking.disambiguator.prior_only import PriorDisambiguator
from retriever.configs import config_utils


class EntityLinker(object):

    def __init__(self, config: Dict[str, Any]):

        if not isinstance(config, Dict):
            config = config_utils.get_config(config)

        self.config = config

        # self.mention_detector = NERMentionDetector(config)
        self.mention_detector = WebAPINERMentionDetector(config)

        self.entity_disambiguator = PriorDisambiguator(config)

    def predict(self, sentence, topk=3):
        mentions = self.mention_detector.predict(sentence)
        ed_output = self.entity_disambiguator.predict(mentions, topk=topk)
        return ed_output
