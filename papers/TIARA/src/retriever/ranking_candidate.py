# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from utils.config import grailqa_rng_ranking_dev_path, grailqa_rng_ranking_train_path, grailqa_rng_ranking_test_path


class LogicalFormRetriever:
    def __init__(self, train_file_path=grailqa_rng_ranking_train_path, dev_file_path=grailqa_rng_ranking_dev_path, test_file_path=grailqa_rng_ranking_test_path):
        self.data = dict()
        if train_file_path is not None:
            with open(train_file_path, 'r') as file:
                self.train_data = json.load(file)
            self.data.update(self.train_data)
        if dev_file_path is not None:
            with open(dev_file_path, 'r') as file:
                self.dev_data = json.load(file)
            self.data.update(self.dev_data)
        if test_file_path is not None:
            with open(test_file_path, 'r') as file:
                self.test_data = json.load(file)
            self.data.update(self.test_data)

    def get_logical_form_by_question_id(self, question_id, top_k=5):
        item = self.data.get(str(question_id), None)
        if item is None:
            return None
        res = []
        for query in item['candidates']:
            lf = query['logical_form']
            res.append(lf)
        return res[:top_k]


if __name__ == '__main__':
    logical_form_retriever = LogicalFormRetriever()
