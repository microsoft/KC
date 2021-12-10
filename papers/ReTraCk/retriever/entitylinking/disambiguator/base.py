# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Any


class EntityDisambiguator(object):

    def __init__(self, config):
        self.config = config

    def predict(self, mentions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input a dictionary including two fields:
            tokens: e.g., ["hello", "world]
            mentions: e.g., [{"start": 0, "end": 1, "type"}]

        Output Linked Entities results

        Parameters
        ----------
        mentions: Dict[str, Any]

        Returns: Dict[str, Any]
        -------

        """
        return dict()

    def train(self):
        pass

    def eval(self):
        pass

    def from_pretrained(self):
        pass