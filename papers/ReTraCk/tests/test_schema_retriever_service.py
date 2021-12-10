# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import unittest
from paths import root_path
from retriever.schema_retriever.interface import DenseSchemaRetriever
import retriever.schema_retriever.dense_retriever.blink as blink


class TestOnlineService(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(root_path, "./configs/schema_service_config.json")
        self.loaded_retriever = DenseSchemaRetriever(config_path)

    def test(self):
        request = json.loads(
            "{\"question\": \"what conference was held at back bay events center\", \"world\": \"GrailQA\"}")

        cls_res, rel_res = self.loaded_retriever.dense_api(request['question'], request['world'])

        assert True


if __name__ == '__main__':
    unittest.main()
