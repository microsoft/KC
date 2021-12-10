# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

mode = 'dev'  # or train/test
dataset = 'GrailQA'

work_dir = '.'

f = open(f'./Dataset/{dataset}/original/{mode}_v1.json', 'r', encoding='utf8')
data_l = json.load(f)

cnt = 0
cnt_relation=0
cnt_class=0

class_pred_file = f'{work_dir}/Class.res'
rel_pred_file = f'{work_dir}/Relation.res'

res = {}

relation_pres = []
class_pres = []
k_rel = 150
k_cls = 100

class_preds = json.load(open(class_pred_file, 'r', encoding='utf8'))
for pred in class_preds:
    curr_pre = pred["predictions"].split('\t')
    curr_pre_score = pred["scores"].split('\t')
    tmp = []
    for n in range(k_rel):
        tmp.append((curr_pre[n], curr_pre_score[n]))
    class_pres.append(tmp)

rel_preds = json.load(open(rel_pred_file, 'r', encoding='utf8'))
for pred in rel_preds:
    curr_pre = pred["predictions"].split('\t')
    curr_pre_score = pred["scores"].split('\t')
    tmp = []
    for n in range(k_cls):
        tmp.append((curr_pre[n], curr_pre_score[n]))
    relation_pres.append(tmp)

gold_ans = []
f_res = open(f'{work_dir}/{dataset}_{mode}_res_Rel{k_rel}_Cls{k_cls}.jsonl', 'w', encoding='utf8')
f_gold = open(f'{work_dir}/{dataset}_{mode}_gold_ans', 'w', encoding='utf8')

for d_l in data_l[:len(relation_pres)]:
    exp = {
        "qid": d_l["qid"],
        "classes": class_pres[cnt],
        "relations": relation_pres[cnt],
    }

    f_res.write(json.dumps(exp) + "\n")
    relations = []
    classes = []

    for nodes in d_l["graph_query"]["edges"]:
        relations.append(nodes["relation"])

    for nodes in d_l["graph_query"]["nodes"]:
        if nodes["node_type"] == "class":
            classes.append(nodes["id"])

    golds = {
        "relations": relations,
        "classes": classes
    }

    gold_ans.append(golds)

    cnt += 1

json.dump(gold_ans, f_gold)
f_gold.close()
f_res.close()
