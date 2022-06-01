'''
The code is adapted from https://github.com/tdopierre/ProtAugment/blob/main/utils/data.py
'''

import numpy as np
import random
import collections
import os
import json
from typing import List, Dict

def get_jsonl_data(jsonl_path: str):
    assert jsonl_path.endswith(".jsonl")
    out = list()
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for line in file:
            j = json.loads(line.strip())
            out.append(j)
    return out

def write_jsonl_data(jsonl_data: List[Dict], jsonl_path: str, force=False):
    if os.path.exists(jsonl_path) and not force:
        raise FileExistsError
    with open(jsonl_path, 'w') as file:
        for line in jsonl_data:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')

def raw_data_to_dict(data, shuffle=True):
    labels_dict = collections.defaultdict(list)
    for item in data:
        labels_dict[item['label']].append(item)
    labels_dict = dict(labels_dict)
    if shuffle:
        for key, val in labels_dict.items():
            random.shuffle(val)
    print(list(labels_dict.keys()))
    return labels_dict

class FewShotDataLoader:
    def __init__(self, file_path):
        self.raw_data = get_jsonl_data(file_path)
        self.data_dict = raw_data_to_dict(self.raw_data, shuffle=True)

    def create_episode(self, n_support: int = 0, n_classes: int = 0, n_query: int = 0):
        episode = dict()
        if n_classes:
            n_classes = min(n_classes, len(self.data_dict.keys()))
            rand_keys = np.random.choice(list(self.data_dict.keys()), n_classes, replace=False)

            while min([len(self.data_dict[k]) for k in rand_keys]) < n_support + n_query:
                rand_keys = np.random.choice(list(self.data_dict.keys()), n_classes, replace=False)
            # assert min([len(val) for val in self.data_dict.values()]) >= n_support + n_query + n_unlabeled

            for key, val in self.data_dict.items():
                random.shuffle(val)

            if n_support:
                episode["xs"] = [[self.data_dict[k][i] for i in range(n_support)] for k in rand_keys]
            if n_query:
                episode["xq"] = [[self.data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys]

        return episode, rand_keys
