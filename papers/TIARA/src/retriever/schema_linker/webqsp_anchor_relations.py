# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('retriever')
import os
from tqdm import tqdm
from dataloader.webqsp_json_loader import WebQSPJsonLoader
from utils.cached_enumeration import FBTwoHopPathCache
from retriever.entity_linker.webqsp_entity_linker import WebQSPEntityLinker
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import webqsp_train_path, webqsp_test_path, webqsp_question_2hop_relation_path
from utils.file_util import pickle_save
from utils.metrics import Metrics, get_recall


def anchor_relations_statistics(dataloader):
    all_relations = set()
    metrics = Metrics()
    for idx in tqdm(range(0, dataloader.len)):
        qid = dataloader.get_question_id_by_idx(idx)
        if dataloader.get_dataset_split() in ['train', 'dev']:
            mid = dataloader.get_golden_entity_mid_by_idx(idx)
        else:
            mid = entity_linker.get_entities_by_question_id(qid, only_id=True)
        golden_relations = dataloader.get_golden_relation_by_idx(idx)

        retrieved_relations = set()
        for entity in mid:
            if entity is None or len(entity) == 0:
                continue
            try:
                entity_relations = two_hop_path_cache.query_two_hop_relations(entity)
                if entity_relations is not None:
                    retrieved_relations.update(entity_relations)
            except Exception as e:
                print('Exception: get 2-hop relations of entity {} failed'.format(entity))

        recall = get_recall(golden_relations, retrieved_relations)
        metrics.add_metric('recall', recall)
        metrics.count()
        all_relations.update(retrieved_relations)
        question_relations_map[qid] = retrieved_relations

        print('{}/{}, len: {}, recall:{}'.format(idx + 1, dataloader.len, len(retrieved_relations), metrics.get_metric('recall')))

    print('dataset overall recall: {}'.format(get_recall(dataloader.get_golden_relations(), all_relations)))


if __name__ == '__main__':
    if os.path.isfile(webqsp_question_2hop_relation_path):
        print('question_relations_map already exists, skip')
    else:
        print('question_relations_map building...')
        retriever = FreebaseRetriever()
        two_hop_path_cache = FBTwoHopPathCache()
        two_hop_path_cache.DATASET = 'webqsp'
        webqsp_train_data = WebQSPJsonLoader(webqsp_train_path)
        webqsp_test_data = WebQSPJsonLoader(webqsp_test_path)
        entity_linker = WebQSPEntityLinker(retriever)

        question_relations_map = dict()

        anchor_relations_statistics(webqsp_train_data)
        anchor_relations_statistics(webqsp_test_data)

        pickle_save(question_relations_map, webqsp_question_2hop_relation_path)
        print('question_relations_map saved to {}'.format(webqsp_question_2hop_relation_path))
