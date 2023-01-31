# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.entity_linker.grailqa_entity_linker import GrailQAEntityLinker
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import freebase_cache_dir
from utils.file_util import pickle_save
from utils.uri_util import remove_ns, ns


def check_dict(d, dict_name):
    for key in d:
        if d[key] is None or d[key] == '':
            print(dict_name + ' key: ' + key)


def check_cache():
    retriever = FreebaseRetriever()
    check_dict(retriever.names, 'names')
    check_dict(retriever.alias, 'alias')
    check_dict(retriever.forward_relations, 'forward relation')
    check_dict(retriever.backward_relations, 'backward relation')

    for key in retriever.forward_neighbors:
        if retriever.forward_neighbors[key] is None or len(retriever.forward_neighbors[key]) == 0:
            print('forward neighbor key: ' + key + ', ' + str(retriever.forward_neighbors[key]))
            continue
        for key1 in retriever.forward_neighbors[key]:
            if retriever.forward_neighbors[key][key1] is None or len(retriever.forward_neighbors[key][key1]) == 0:
                print(
                    'forward neighbor key: (' + key + ', ' + key1 + '), ' + str(retriever.forward_neighbors[key][key1]))

    for key in retriever.backward_neighbors:
        if retriever.backward_neighbors[key] is None or len(retriever.backward_neighbors[key]) == 0:
            print('backward neighbor key: ' + key + ', ' + str(retriever.backward_neighbors[key]))
            continue
        for key1 in retriever.backward_neighbors[key]:
            if retriever.backward_neighbors[key][key1] is None or len(retriever.backward_neighbors[key][key1]) == 0:
                print('backward neighbor key: (' + key + ', ' + key1 + '), ' + str(
                    retriever.backward_neighbors[key][key1]))


def cache_var(var, forward_relations, backward_relations, names, alias, golden_sparql, retriever, white_list=None):
    if var not in golden_sparql:
        return
    sparql = re.sub(' +', ' ', golden_sparql).replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ' + var)
    values = retriever.query_var(sparql, var)
    for value in values:
        rn_value = remove_ns(value)
        if rn_value not in forward_relations:
            forward_relations[rn_value] = set(
                retriever.relation_value_by_mid_list(value, forward=True, backward=False,
                                                     white_list=white_list))
        if rn_value not in backward_relations:
            backward_relations[rn_value] = set(
                retriever.relation_value_by_mid_list(value, forward=False, backward=True,
                                                     white_list=white_list))
        if rn_value not in names:
            names[rn_value] = retriever.entity_name_by_mid(value)
        if rn_value not in alias:
            alias[rn_value] = retriever.entity_alias_by_mid(value)


def create_neighbor_cache(output_dir=freebase_cache_dir):
    retriever = FreebaseRetriever(cache_dir=freebase_cache_dir)
    forward_neighbors = dict()
    backward_neighbors = dict()

    count = 0
    if retriever.forward_relations is not None:
        for mid in retriever.forward_relations.keys():
            if mid not in forward_neighbors:
                forward_neighbors[mid] = dict()
            for r in retriever.forward_relations[mid]:
                r = remove_ns(r)
                if r not in forward_neighbors[mid]:
                    try:
                        neighbors = retriever.neighbor_value_by_mid_and_relation(ns(mid), ns(r),
                                                                                 forward=True,
                                                                                 backward=False)
                        if len(neighbors) != 0:
                            forward_neighbors[mid][r] = neighbors
                    except Exception as e:
                        print(e)
            if count % 50 == 0:
                print('[INFO] forward neighbors: ' + str(count) + ' / ' + str(len(retriever.forward_relations)))
            count += 1
        print('\n')

    count = 0
    if retriever.backward_relations is not None:
        for mid in retriever.backward_relations.keys():
            if mid not in backward_neighbors:
                backward_neighbors[mid] = dict()
            for r in retriever.backward_relations[mid]:
                r = remove_ns(r)
                if r not in backward_neighbors[mid]:
                    try:
                        neighbors = retriever.neighbor_value_by_mid_and_relation(ns(mid), ns(r),
                                                                                 forward=False,
                                                                                 backward=True)
                        if len(neighbors) != 0:
                            backward_neighbors[mid][r] = neighbors
                    except Exception as e:
                        print(e)
            if count % 50 == 0:
                print('[INFO] backward neighbors: ' + str(count) + ' / ' + str(len(retriever.backward_relations)))
            count += 1
        print('\n')

    pickle_save(forward_neighbors, output_dir + '/forward_nei.bin')
    pickle_save(backward_neighbors, output_dir + '/backward_nei.bin')


def create_rdf_type_cache():
    retriever = FreebaseRetriever()
    res = dict()
    count = 0

    file_path_list = ['../dataset/GrailQA/grailqa_v1.0_train.json',
                      '../dataset/GrailQA/grailqa_v1.0_dev.json',
                      '../dataset/GrailQA/grailqa_el.json']
    for file_path in file_path_list:
        count = 0
        uri_set = None
        if 'grailqa_v' in file_path:
            dataloader = GrailQAJsonLoader(file_path)
            uri_set = dataloader.get_uris()
        elif 'grailqa_el' in file_path:
            linker = GrailQAEntityLinker(file_path)
            uri_set = linker.get_mid_set()

        for uri in uri_set:
            if uri not in res:
                res[uri] = retriever.rdf_type_by_uri(uri)

            if count % 100 == 0:
                print('[INFO] ' + file_path + ' rdf type: ' + str(count) + ' / ' + str(len(uri_set)))
            count += 1

    pickle_save(res, freebase_cache_dir + '/rdf_type.bin')


if __name__ == "__main__":
    # create_relation_cache()
    # create_neighbor_cache()
    # create_predicate_range_cache()
    # check_cache()
    create_rdf_type_cache()
