# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

mode = 'dev'
dataset = 'GrailQA'
dataset_path = f'\\\\msra-km-comp-02\\ReTraCkDemoFiles\\OriginalData\\GrailQA_v1.0\\grailqa_v1.0_{mode}.json'
out_dir = '.'

f_test = open(dataset_path, 'r', encoding='utf-8')
data_l_test = json.load(f_test)


def get_test_data(data_l, f_rel):
    dt = []
    count = 0
    for d_l in data_l:
        mention = d_l["question"]
        label_id = -1
        exp = {
            "id": count,
            "label": "",
            "label_id": label_id,
            "context_left": "".lower(),
            "mention": mention,
            "context_right": "".lower(),
        }
        dt.append(exp)
        count += 1

    print("Count: ", count)
    dt_rel_f = open(f_rel, 'w')
    json.dump(dt, dt_rel_f)


get_test_data(data_l_test, os.path.join(out_dir, f"./noBatch_{dataset}_{mode}.test"))
