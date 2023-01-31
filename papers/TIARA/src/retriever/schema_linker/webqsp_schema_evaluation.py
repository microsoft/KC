# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.webqsp_json_loader import WebQSPJsonLoader
from retriever.schema_linker.schema_file_reader import SchemaLinker
from utils.config import webqsp_train_path, webqsp_test_path
from utils.metrics import Metrics, get_precision, get_recall, get_f1_score_by_pr


def schema_eval(schema_linker, dataloader, verbose=False):
    metrics = Metrics()
    for idx in range(0, dataloader.len):
        qid = dataloader.get_question_id_by_idx(idx)
        golden_relations = dataloader.get_question_predicates_by_idx(idx)
        predicted_relations = schema_linker.get_relation_by_question_id(qid)

        precision = get_precision(golden_relations, predicted_relations)
        recall = get_recall(golden_relations, predicted_relations)
        f1 = get_f1_score_by_pr(precision, recall)

        if recall < 1 and verbose:
            print('[qid] ' + str(qid))
            print('[Golden] ' + str(golden_relations))
            print('[Predicted] ' + str(predicted_relations))
            print('recall: ' + str(recall))
            print()
        metrics.add_metric('relation P', precision)
        metrics.add_metric('relation R', recall)
        metrics.add_metric('relation F', f1)
        metrics.count()
    print(metrics.get_metrics(['relation P', 'relation R', 'relation F']))


if __name__ == '__main__':
    webqsp_train = WebQSPJsonLoader(webqsp_train_path)
    webqsp_test = WebQSPJsonLoader(webqsp_test_path)

    retrack_linker = SchemaLinker(train_file_path='../dataset/WebQSP/schema_linking_results/webQSPdenseEmbeddingTrainRes.jsonl',
                                  dev_file_path=None,
                                  test_file_path='../dataset/WebQSP/schema_linking_results/webQSPdenseEmbeddingTestRes.jsonl')

    linker = SchemaLinker(train_file_path=None,
                          dev_file_path='../logs/schema_retrieval/webqsp_dev_relations_0525.json',
                          test_file_path='../logs/schema_retrieval/webqsp_test_relations_0525.json', file_format='json')

    schema_eval(retrack_linker, webqsp_train)
    schema_eval(retrack_linker, webqsp_test)
    schema_eval(linker, webqsp_test)
