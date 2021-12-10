# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests
import json
from typing import Dict
from retriever.configs import config_utils
from retriever.schema_retriever.interface import DenseSchemaRetriever


class DenseSchemaRetrieverClient(object):

    def __init__(self, config):

        if not isinstance(config, Dict):
            config = config_utils.get_config(config)

        self.config = config

        if self.config["schema_query_mode"] == 'offline':
            self.schema_retriever = DenseSchemaRetriever(config["schema_service_config_path"])


    def predict_local(self, sentence, world):

        # output = self.schema_retriever.predict(sentence, world)
        output = self.schema_retriever.dense_api(sentence, self.config['world'])
        return output

    def predict_remote(self, sentence, world):

        data = {
            "question": sentence,
            "world": world
        }

        # print('Querying: ' + self.uri)

        try:
            uri = self.config["dense_retriever_uri"]
            output = requests.post(uri, json=data)
        except Exception as e:
            print('Exception interacting with: ' + uri)
            print(str(e))
            raise Exception("Error talking to: " + uri)

        return output

    def predict(self, sentence, world="GrailQA"):

        if self.config["schema_query_mode"] == 'online':
            output = self.predict_remote(sentence, world)
            json_output = json.loads(output.text)
            types = [(x[0], x[1]) for x in json_output["Candidate classes"]]
            relations = [(x[0], x[1]) for x in json_output["Candidate Relations"]]
        else:
            output = self.predict_local(sentence, world)
            types, relations = output

        return types, relations


if __name__ == "__main__":

    default_config = {
        "dense_retriever_uri": "http://localhost:6200/api/schema"  # Default value, should be specified in config
    }
    tester = DenseSchemaRetrieverClient(default_config)
    print(tester.predict("hello world", "WebQSP"))
