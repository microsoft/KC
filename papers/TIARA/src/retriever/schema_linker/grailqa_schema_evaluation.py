# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
from dataloader.grailqa_json_loader import GrailQAJsonLoader
from utils.config import grailqa_dev_path, grailqa_train_path
from utils.metrics import get_recall
from retriever.schema_linker.schema_file_reader import SchemaLinker


def schema_eval(schema_linker, dataloader):
    metrics = dict()
    for idx in range(0, dataloader.len):
        qid = dataloader.get_question_id_by_idx(idx)
        level = dataloader.get_level_by_idx(idx)
        golden_classes = dataloader.get_golden_class_by_idx(idx, True)
        golden_relations = dataloader.get_golden_relation_by_idx(idx)

        predicted_classes = schema_linker.get_class_by_question_id(qid)
        predicted_relations = schema_linker.get_relation_by_question_id(int(qid))

        class_recall = get_recall(golden_classes, predicted_classes)
        relation_recall = get_recall(golden_relations, predicted_relations)
        metrics['class recall'] = metrics.get('class recall', 0) + class_recall
        metrics['class recall ' + level] = metrics.get('class recall ' + level, 0) + class_recall
        metrics['relation recall'] = metrics.get('relation recall', 0) + relation_recall
        metrics['relation recall ' + level] = metrics.get('relation recall ' + level, 0) + relation_recall
        metrics[level] = metrics.get(level, 0) + 1
    split = dataloader.get_dataset_split()
    print(split + ' class recall', metrics.get('class recall', 0) / dataloader.len)
    if split != 'train':
        print(split + ' i.i.d.', metrics.get('class recall i.i.d.', 0) / metrics.get('i.i.d.', 1))
        print(split + ' compositional', metrics.get('class recall compositional', 0) / metrics.get('compositional', 1))
        print(split + ' zero-shot', metrics.get('class recall zero-shot', 0) / metrics.get('zero-shot', 1))
    print(split + ' relation recall', metrics.get('relation recall', 0) / dataloader.len)
    if split != 'train':
        print(split + ' i.i.d.', metrics.get('relation recall i.i.d.', 0) / metrics.get('i.i.d.', 1))
        print(split + ' compositional', metrics.get('relation recall compositional', 0) / metrics.get('compositional', 1))
        print(split + ' zero-shot', metrics.get('relation recall zero-shot', 0) / metrics.get('zero-shot', 1))
    print()


if __name__ == '__main__':
    grailqa_train = GrailQAJsonLoader(grailqa_train_path)
    grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)

    linker = SchemaLinker(train_file_path='../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_train.jsonl',
                          dev_file_path='../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev.jsonl')

    schema_eval(linker, grailqa_train)
    schema_eval(linker, grailqa_dev)
