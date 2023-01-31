# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.webqsp_json_loader import WebQSPJsonLoader
from utils.config import webqsp_train_path, webqsp_test_path, webqsp_train_gen_path, webqsp_test_gen_path
from utils.file_util import read_json_file


class WebQSPGenerationData:
    def __init__(self, train_file_path=webqsp_train_path,
                 train_gen_file_path=webqsp_train_gen_path,
                 test_file_path=webqsp_test_path,
                 test_gen_file_path=webqsp_test_gen_path):
        self.generation_target = dict()  # question id -> generation target
        self.train_gen = read_json_file(train_gen_file_path)
        self.test_gen = read_json_file(test_gen_file_path)

        for item in self.train_gen:
            self.generation_target[item['qid']] = item['generation_target']
        for item in self.test_gen:
            self.generation_target[item['qid']] = item['generation_target']

        webqsp_train_data = WebQSPJsonLoader(train_file_path)
        webqsp_test_data = WebQSPJsonLoader(test_file_path)

        self.generation_target_from_dataset(webqsp_train_data)
        self.generation_target_from_dataset(webqsp_test_data)

    def generation_target_from_dataset(self, data):
        for idx in range(0, data.len):
            qid = data.get_question_id_by_idx(idx)
            if qid in self.generation_target and self.generation_target[qid] != 'null':
                continue
            s_expression = data.get_s_expression_by_idx(idx)
            if s_expression is None or s_expression == 'null':
                continue
            self.generation_target[qid] = s_expression

    def get_generation_target_by_question_id(self, question_id: str):
        return self.generation_target.get(question_id, None)
