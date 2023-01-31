# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys

sys.path.append('.')
import json

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.entity_linker.grailqa_entity_linker import GrailQAEntityLinker
from utils.config import grailqa_dev_path, grailqa_entity_linking_path, grailqa_rng_el_path, grailqa_tiara_dev_el_path, grailqa_train_path
from utils.metrics import Metrics, get_precision, get_recall, get_f1_score_by_pr


def facc1_statistics(data: GrailQAJsonLoader):
    with open('../logs/rng_logs/grail_dev_entities.json', 'r') as f:
        facc1_entities = json.load(f)

    metrics = Metrics()
    for idx in range(0, data.len):
        qid = str(data.get_question_id_by_idx(idx))
        golden_entities = data.get_golden_entity_by_idx(idx, only_id=True)

        if len(golden_entities) == 0:
            continue
        pred_entities = []

        if qid in facc1_entities:
            pred = facc1_entities[qid]
            for entity_list in pred:
                pred_entities += [item['id'] for item in entity_list]

        p = get_precision(golden_entities, pred_entities)
        r = get_recall(golden_entities, pred_entities)
        f1 = get_f1_score_by_pr(p, r)
        metrics.add_metric('precision', p)
        metrics.add_metric('recall', r)
        metrics.add_metric('F1', f1)
        metrics.count()
        # end for each question
    print(metrics.get_metrics(['precision', 'recall', 'F1']))


def statistics(entity_linker, data: GrailQAJsonLoader):
    metrics = Metrics()
    metrics_level = {}
    for idx in range(data.len):
        qid = data.get_question_id_by_idx(idx)
        golden_entities = data.get_golden_entity_by_idx(idx, only_id=True)
        pred_entities = entity_linker.get_entities_by_question_id(qid, only_id=True)
        level = data.get_level_by_idx(idx)

        if len(golden_entities) == 0 and len(pred_entities) == 0:
            p = r = f1 = 1.0
        elif len(golden_entities) == 0 and len(pred_entities) != 0:
            p = 0.0
            r = 1.0
            f1 = 0.0
        elif len(golden_entities) != 0 and len(pred_entities) == 0:
            p = 1.0
            r = 0.0
            f1 = 0.0
        else:
            p = get_precision(golden_entities, pred_entities)
            r = get_recall(golden_entities, pred_entities)
            f1 = get_f1_score_by_pr(p, r)
        metrics.add_metric('P', p)
        metrics.add_metric('R', r)
        metrics.add_metric('F1', f1)
        if level != 'n/a':
            metrics_level[level + '-P'] = metrics_level.get(level + '-P', 0) + p
            metrics_level[level + '-R'] = metrics_level.get(level + '-R', 0) + r
            metrics_level[level + '-F1'] = metrics_level.get(level + '-F1', 0) + f1
            metrics_level[level] = metrics_level.get(level, 0) + 1
        metrics.count()
    print(metrics.get_metrics(['P', 'R', 'F1']))
    for key in ['i.i.d.', 'compositional', 'zero-shot']:
        if key in metrics_level and metrics_level[key] != 0:
            print(key, 'P', metrics_level[key + '-P'] / metrics_level[key],
                  'R', metrics_level[key + '-R'] / metrics_level[key],
                  'F1', metrics_level[key + '-F1'] / metrics_level[key])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dev')
    parser.add_argument('--el_path', type=str, default=grailqa_tiara_dev_el_path)
    args = parser.parse_args()

    if args.data == 'dev':
        data = GrailQAJsonLoader(grailqa_dev_path)
    elif args.data == 'train':
        data = GrailQAJsonLoader(grailqa_train_path)
    elif os.path.isfile(args.data):
        data = GrailQAJsonLoader(args.data)

    tiara_entity_linker = GrailQAEntityLinker(args.el_path)
    statistics(tiara_entity_linker, data)
