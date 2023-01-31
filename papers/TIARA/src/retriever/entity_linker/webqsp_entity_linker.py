# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.config import webqsp_train_elq_path, webqsp_test_elq_path, webqsp_train_oracle_entity_path, webqsp_test_oracle_entity_path
from utils.file_util import read_json_file


class WebQSPEntityLinker:
    def __init__(self, retriever,
                 oracle_train_file_path=webqsp_train_oracle_entity_path, oracle_test_file_path=webqsp_test_oracle_entity_path,
                 train_file_path=webqsp_train_elq_path, test_file_path=webqsp_test_elq_path):
        self.retriever = retriever

        self.oracle = dict()
        oracle_train_entities = read_json_file(oracle_train_file_path)
        oracle_test_entities = read_json_file(oracle_test_file_path)
        for item in oracle_train_entities:
            self.oracle[item['id']] = item
        for item in oracle_test_entities:
            self.oracle[item['id']] = item

        self.linker = dict()
        train_entities = read_json_file(train_file_path)
        test_entities = read_json_file(test_file_path)
        for item in train_entities:
            self.linker[item['id']] = item
        for item in test_entities:
            self.linker[item['id']] = item

    def get_entities_by_question_id(self, question_id: str, only_id=False, lower=True, oracle=False):
        question_id = str(question_id)
        res = []
        if oracle is False and question_id in self.linker:
            res = self.linker[question_id]
        elif oracle is True and question_id in self.oracle:
            res = self.oracle[question_id]

        if only_id:
            return list(set(res['freebase_ids']))
        else:
            res_new = []
            entity_set = set()
            for ent_idx in range(len(res['freebase_ids'])):
                mid = res['freebase_ids'][ent_idx]
                if mid in entity_set:
                    continue
                entity_set.add(mid)
                res_new.append({'id': mid, 'friendly_name': res['pred_tuples_string'][ent_idx][0].lower()})
            return res_new
