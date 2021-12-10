# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from allennlp.common.testing import AllenNlpTestCase
from retriever.entitylinking.bert_ner.bert import NER


class TestBERTNER(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.ner = NER("path_to_ner_model_path")

    def test_case1(self):
        output = self.ner.run_ner("what is the name of the film festival that concentrates on short films ?")
        print(output)
