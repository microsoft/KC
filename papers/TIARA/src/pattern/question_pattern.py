# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.stanza_util import StanzaService, get_dependency_sequence, get_dependency_sequence_by_parsing
from utils.string_util import string_replace


def get_question_with_candidate_query(question, candidate_queries: list, query_norm=False):
    res = question
    if candidate_queries is not None:
        if query_norm is False:
            res += '|query'
            for candidate_query in candidate_queries:
                res += '|' + candidate_query
        elif query_norm is True:
            for candidate_query in candidate_queries:
                res += '; ' + candidate_query
    return res


def get_question_with_entity_and_schema(question, entities, relations, classes, entity_mid=False, query_norm=False):
    res = question
    if query_norm is False and entities is not None and len(entities) > 0:
        res += '|entity'
        for e in entities:
            if 'friendly_name' in e:
                res += '|' + e['friendly_name'].lower()
            elif 'mention' in e:
                res += '|' + e['mention'].lower()
            elif 'label' in e:
                res += '|' + e['label'].lower()
            else:
                continue
            if entity_mid:
                res += ' ' + e['id']

    if classes is not None and len(classes) > 0:
        res += '|class'
        for c in classes:
            res += '|' + c

    if relations is not None and len(relations) > 0:
        res += '|relation'
        for r in relations:
            res += '|' + r

    return res
