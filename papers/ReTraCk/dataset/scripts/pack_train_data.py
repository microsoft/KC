# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

f = open('grailqa_v1.0_train.json', 'r')
data_l = json.load(f)

f_rel_id = open('freebase_relation_id', 'r')
rel2id = json.load(f_rel_id)

f_cls_id = open('freebase_class_id', 'r')
cls2id = json.load(f_cls_id)

"""
Formatted like this:
{
    "id": 0,
    "label": "unknown",
    "label_id": -1,
    "context_left": "".lower(),
    "mention": "Shakespeare".lower(),
    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
}
"""


# For training, sample some training data as dev set
def get_boBatch_data(data_l):
    rel_res = {}
    cls_res = {}
    for i in range(10):
        rel_res[str(i)] = []
        cls_res[str(i)] = []

    dt_rel = []
    dt_cls = []
    dd_rel = []
    dd_cls = []

    cnt_rel = 0
    cnt_cls = 0
    cnt = 0
    wrong_rel = 0
    wrong_cls = 0
    for d_l in data_l:
        cnt += 1
        mention = d_l["question"]
        label_id = -1
        tmp_rel = 0
        tmp_cls = 0
        for items_l in d_l["graph_query"]["edges"]:
            items = items_l["relation"]

            try:
                label_id = rel2id[items]
            except:
                wrong_rel += 1
                print("rel_items", items)

            exp = {
                "id": cnt_rel,
                "label": items.replace('.', ' ').replace('_', ' '),
                "label_id": label_id,
                "context_left": "".lower(),
                "mention": mention,
                "context_right": "".lower(),
            }
            rel_res[str(tmp_rel)].append(exp)
            tmp_rel += 1

        for items_l in d_l["graph_query"]["nodes"]:
            if items_l["node_type"] == "class":
                items = items_l["id"]
                try:
                    label_id = cls2id[items]
                except:
                    wrong_cls += 1
                    print("cls_items", items)

                exp = {
                    "id": cnt_cls,
                    "label": items.replace('.', ' ').replace('_', ' '),
                    "label_id": label_id,
                    "context_left": "".lower(),
                    "mention": mention,
                    "context_right": "".lower(),
                }
                cls_res[str(tmp_cls)].append(exp)
                tmp_cls += 1

    for k, v in rel_res.items():
        for exp in v:
            exp["id"] = cnt_rel
            cnt_rel += 1
            if cnt_rel % 100 == 0:
                dd_rel.append(exp)
            else:
                dt_rel.append(exp)

    for k, v in cls_res.items():
        for exp in v:
            exp["id"] = cnt_cls
            cnt_cls += 1
            if cnt_cls % 100 == 0:
                dd_cls.append(exp)
            else:
                dt_cls.append(exp)

    return dt_rel, dt_cls, dd_rel, dd_cls


dt_rel, dt_cls, dd_rel, dd_cls = get_boBatch_data(data_l)

frelation = open("train_all_rel.train", 'w')
json.dump(dt_rel, frelation)
fcls = open("train_all_cls.train", 'w')
json.dump(dt_cls, fcls)

frelation = open("train_all_rel.dev", 'w')
json.dump(dd_rel, frelation)
fcls = open("train_all_cls.dev", 'w')
json.dump(dd_cls, fcls)
