# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from utils.config import grailqa_train_path


def is_number(s: str):
    s = s.strip('\'\"')
    if s.startswith('-'):
        s = s[1:]
    if s.isdigit():
        return True
    try:
        num = float(s)
        return True
    except ValueError:
        return False


class GrailQAEntityLinker:

    def __init__(self, dev_file_path: str, train_path=grailqa_train_path, test_file_path=None):
        self.data = {}
        with open(dev_file_path, 'r', encoding='UTF-8') as file:
            f = json.load(file)
            self.data = f
            self.len = len(self.data)
        if test_file_path is not None:
            with open(test_file_path, 'r', encoding='UTF-8') as file:
                f = json.load(file)
                self.data.update(f)
                self.len += len(f)

        self.train_data = GrailQAJsonLoader(train_path)

    def get_len(self):
        return self.len

    def get_entities_by_question_id(self, question_id, only_id=False) -> list:
        if isinstance(question_id, int):
            question_id = str(question_id)
        res = []
        idx = self.train_data.get_idx_by_question_id(question_id)
        if idx != -1:  # question in training data
            for node in self.train_data.get_graph_query(idx)['nodes']:
                if node['node_type'] == 'entity':
                    res.append(node)
            return res
        else:  # question in validation / testing data
            if question_id not in self.data:
                return res
            res = self.data[question_id]['entities']

        if len(res):
            for mid in res:
                if 'friendly_name' in res[mid]:
                    key = 'friendly_name'
                elif 'mention' in res[mid]:
                    key = 'mention'
                elif 'label' in res[mid]:
                    key = 'label'
                break
        test_dict_list = sorted(list(res.items()), key=lambda item: len(item[1][key]), reverse=True)

        res = []
        for item in test_dict_list:
            item[1]['id'] = item[0]
            res.append(item[1])

        if only_id:
            return [item['id'] for item in res]
        return res

    def get_entity_mentions_by_question_id(self, question_id):
        res = set()
        if str(question_id) not in self.data:  # question in training data
            idx = self.train_data.get_idx_by_question_id(question_id)
            assert idx != -1
            for node in self.train_data.get_graph_query(idx)['nodes']:
                if node['node_type'] == 'entity' and is_number(node['friendly_name']) is False:
                    res.add(node['friendly_name'].replace('  ', ' ').lower())
        else:  # question in validation / testing data
            for mid in self.data[str(question_id)]['entities']:
                mention = self.data[str(question_id)]['entities'][mid]['mention']
                if is_number(mention) is False:
                    res.add(mention)
        res = list(res)
        res.sort(key=lambda x: len(x), reverse=True)
        return res

    def get_entity_friendly_name_by_question_id(self, question_id: str):
        res = []
        for mid in self.data[question_id]['entities']:
            res.append(self.data[question_id]['entities'][mid]['friendly_name'])
        return res

    def get_mid_set(self):
        res = set()
        for question_id in self.data:
            for mid in self.data[question_id]['entities']:
                res.add(mid)
        return res
