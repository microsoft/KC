# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import json
import requests
from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader
from typing import Optional, Dict, List, Set
import pickle

from utils import RegisterWorld

DEFAULT_RETRIEVER_END_POINT = "http://localhost:6100/kbqa/api/v1.0/retriever"


@Predictor.register("kbqa")
class KBQAPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)
        if self._dataset_reader.semantic_constraint_file:
            constraint_f = open(self._dataset_reader.semantic_constraint_file, "rb")
            self.all_relation_dict: Optional[Dict[str, List]] = pickle.load(constraint_f)
        else:
            self.all_relation_dict = None

        if self._dataset_reader.literal_relation_file:
            literal_f = open(self._dataset_reader.literal_relation_file, "rb")
            self.all_litearl_relation: Optional[Set] = pickle.load(literal_f)
        else:
            self.all_litearl_relation = None

        # Default world is GrailQA, will be changed when loading service
        self.world = RegisterWorld.GrailQA
        self.schema_retriever_uri = DEFAULT_RETRIEVER_END_POINT

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        sentence = json_dict["sentence"].lower()
        json_dict["world"] = self.world

        print("Querying retriever at: " + self.schema_retriever_uri)
        response = requests.post(self.schema_retriever_uri, json=json_dict)

        # check response

        example = json.loads(response.content.decode())

        # construct a new example
        example["sentence"] = sentence
        example["qid"] = "test"
        data_sample = self._dataset_reader.build_data_sample(ex=example, key="graph_query")
        return self._dataset_reader.text_to_instance(semantic_cons=None,
                                                     literal_relation=self.all_litearl_relation,
                                                     entity_offset_tokens=example["bert_tokens"],
                                                     is_training=False,
                                                     **data_sample)
