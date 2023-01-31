# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.webqsp_json_loader import WebQSPJsonLoader
from retriever.ranking_candidate import LogicalFormRetriever
from retriever.schema_linker.schema_file_reader import SchemaLinker
from utils.config import webqsp_test_path, webqsp_train_path, webqsp_rng_ranking_train_path, webqsp_rng_ranking_test_path, webqsp_schema_ptrain_path, webqsp_schema_pdev_path, \
    webqsp_schema_test_path
from utils.logic_form_util import get_entity_schema_in_lisp
from utils.metrics import get_recall
from utils.uri_util import schema_set_domain


def relation_statistics(dataloader: WebQSPJsonLoader, log=None):
    total = dict()
    lf_recall = dict()
    dense_recall = dict()
    lf_selected = 0
    dense_selected = 0
    total_pred = 0

    for idx in range(0, dataloader.len):
        qid = dataloader.get_question_id_by_idx(idx)
        inferential_chain = dataloader.get_golden_relation_by_idx(idx)
        constraint_predicates = dataloader.get_constraint_predicate_by_idx(idx)
        order_predicates = dataloader.get_order_predicate_by_idx(idx)
        golden_relations = dataloader.get_question_predicates_by_idx(idx)
        golden_domains = schema_set_domain(golden_relations)

        if log is not None:
            pred_s_expr = log[qid]['predicted_s_expr']
            if pred_s_expr is not None and len(pred_s_expr):
                total_pred += 1
            pred_entities, pred_relations = get_entity_schema_in_lisp(pred_s_expr)

        logical_forms = ranking_candidate.get_logical_form_by_question_id(qid)
        lf_pred_relations = set()
        if logical_forms is not None:
            for lf in logical_forms:
                entities, relations = get_entity_schema_in_lisp(lf)
                for r in relations:
                    if 'macro' in r:
                        print(r)
                        continue
                    lf_pred_relations.add(r)
        dense_pred_relations = schema_linker.get_relation_by_question_id(qid, 10)
        dense_pred_domains = schema_set_domain(dense_pred_relations)
        dense_pred_domains.update({'base', 'common', 'type', 'user'})
        if get_recall(golden_domains, dense_pred_domains) == 0:
            print(qid, golden_domains, dense_pred_domains)

        if len(inferential_chain):
            total['inferential'] = total.get('inferential', 0) + 1
        if len(constraint_predicates):
            total['constraint'] = total.get('constraint', 0) + 1
        if len(order_predicates):
            total['order'] = total.get('order', 0) + 1
        if len(golden_relations):
            total['domain'] = total.get('domain', 0) + 1

        lf_recall['inferential'] = lf_recall.get('inferential', 0) + get_recall(inferential_chain, lf_pred_relations)
        lf_recall['constraint'] = lf_recall.get('constraint', 0) + get_recall(constraint_predicates, lf_pred_relations)
        lf_recall['order'] = lf_recall.get('order', 0) + get_recall(order_predicates, lf_pred_relations)

        dense_recall['inferential'] = dense_recall.get('inferential', 0) + get_recall(inferential_chain, dense_pred_relations)
        dense_recall['constraint'] = dense_recall.get('constraint', 0) + get_recall(constraint_predicates, dense_pred_relations)
        dense_recall['order'] = dense_recall.get('order', 0) + get_recall(order_predicates, dense_pred_relations)
        dense_recall['domain'] = dense_recall.get('domain', 0) + get_recall(golden_domains, dense_pred_domains)

        if log is not None and len(pred_relations):
            lf_selected += len(set(lf_pred_relations).intersection(pred_relations).difference(dense_pred_relations)) / len(pred_relations)
            if dense_pred_relations is not None:
                dense_selected += len(set(dense_pred_relations).intersection(pred_relations).difference(lf_pred_relations)) / len(pred_relations)

    for key in lf_recall:
        lf_recall[key] = lf_recall[key] / total[key]
    for key in dense_recall:
        dense_recall[key] = dense_recall[key] / total[key]

    print('total:', total)
    print('lf_recall:', lf_recall)
    print('dense_recall:', dense_recall)
    if log is not None:
        print('lf_selected:', lf_selected / total_pred)
        print('dense_selected:', dense_selected / total_pred)


if __name__ == '__main__':
    params = dict()
    params['lf_train'] = webqsp_rng_ranking_train_path
    params['lf_test'] = webqsp_rng_ranking_test_path
    params['schema_train'] = webqsp_schema_ptrain_path
    params['schema_dev'] = webqsp_schema_pdev_path
    params['schema_test'] = webqsp_schema_test_path

    ranking_candidate = LogicalFormRetriever(train_file_path=params['lf_train'], dev_file_path=None, test_file_path=params['lf_test'])
    schema_linker = SchemaLinker(train_file_path=params['schema_train'], dev_file_path=params['schema_dev'], test_file_path=params['schema_test'], file_format='json')

    webqsp_train_data = WebQSPJsonLoader(webqsp_train_path)
    webqsp_test_data = WebQSPJsonLoader(webqsp_test_path)

    # log = read_json_as_dict('../logs/webqsp_2022_05_03_13_53_20_log.json', 'qid')

    relation_statistics(webqsp_train_data)
    relation_statistics(webqsp_test_data)
