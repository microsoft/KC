# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest
from paths import root_path
from retriever.schema_retriever.interface import DenseSchemaRetriever


class TestOnlineService(unittest.TestCase):

    def setUp(self):

        config_path = os.path.join(root_path, "./configs/schema_service_config.json")
        self.sr = DenseSchemaRetriever(config_path)

    def test(self):

        target = "ListQA"
        query = "where is sweden"

        rel, cls = self.sr.dense_api(query, target)


if __name__ == '__main__':

    unittest.main()
