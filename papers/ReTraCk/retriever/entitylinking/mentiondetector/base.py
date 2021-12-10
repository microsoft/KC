# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Any



class MentionDetector(object):

    def __init__(self, config: Dict[str, Any]):
        self.config = config

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
        return dict()

    def train(self):
        pass

    def eval(self):
        pass

    def from_pretrained(self):
        pass