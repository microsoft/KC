# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from SPARQLWrapper import SPARQLWrapper

from retriever.kb_retriever import KBRetriever
from utils.config import freebase_addr, freebase_port, grailqa_reverse_properties_dict_path, grailqa_property_roles_dict_path, freebase_cache_dir
from utils.domain_dict import fb_domain_dict
from utils.file_util import pickle_load, pickle_save, read_tsv_as_list
from utils.sparql_generator import SparqlGenerator
from utils.triple import Triple
from utils.uri_util import freebase_instance_uri, remove_ns, ns, is_mid_gid, add_quotation_mark


def get_values_in_list(relation_list, white_list=None):
    res = []
    for relation in relation_list:
        if isinstance(relation, dict):
            relation = relation['value']
        res.append(relation)
    if white_list is not None:
        res_white = []
        for r in res:
            if remove_ns(r) in white_list:
                res_white.append(r)
        res = res_white
    return res


class FreebaseRetriever(KBRetriever):
    # cache dict
    forward_relations: dict
    backward_relations: dict
    names: dict
    alias: dict
    forward_neighbors: dict
    backward_neighbors: dict
    predicate_range_cache: dict
    rdf_type_cache: dict
    reverse_relation: dict
    update_count: dict  # determine if cache has been changed and needs to be saved again

    def __init__(self, cache_dir=freebase_cache_dir, freebase_addr=freebase_addr, freebase_port=freebase_port):
        super().__init__()
        self.kb_name = 'freebase'
        self.sparql_wrapper = SPARQLWrapper('http://' + freebase_addr + ':' + str(freebase_port) + '/sparql')
        # self.sparql_wrapper.setUseKeepAlive()  # https://sparqlwrapper.readthedocs.io/en/latest/SPARQLWrapper.Wrapper.html?highlight=setusekeepalive#SPARQLWrapper.Wrapper.SPARQLWrapper.setUseKeepAlive

        # load caches
        self.roles = []  # a list of tuples (domain, property, range)
        self.forward_relations = pickle_load(cache_dir + '/forward.bin', dict())  # mid -> forward relations
        self.backward_relations = pickle_load(cache_dir + '/backward.bin', dict())  # mid -> backward relations
        self.names = pickle_load(cache_dir + '/names.bin', dict())  # mid -> name
        self.alias = pickle_load(cache_dir + '/alias.bin', dict())  # mid -> alias list
        self.forward_neighbors = pickle_load(cache_dir + '/forward_nei.bin', dict())  # mid -> forward relation -> forward neighbors
        self.backward_neighbors = pickle_load(cache_dir + '/backward_nei.bin', dict())  # mid -> backward relation -> backward neighbors
        self.rdf_type_cache = pickle_load(cache_dir + '/rdf_type.bin', dict())  # uri -> rdf_type
        self.cache_dir = cache_dir
        self.update_count = {}

        if os.path.isfile(grailqa_property_roles_dict_path):
            self.roles = read_tsv_as_list(grailqa_property_roles_dict_path, sep=' ')
            self.predicate_range_cache = dict()
            for t in self.roles:
                self.predicate_range_cache[t[1]] = t[2]

        if os.path.isfile(grailqa_reverse_properties_dict_path):
            pairs = read_tsv_as_list(grailqa_reverse_properties_dict_path)
            self.reverse_relation_cache = dict()
            for pair in pairs:
                self.reverse_relation_cache[pair[0]] = pair[1]

    def relation_by_mid(self, mid: str, s_filter=None, p_filter_list=None, o_filter=None, forward=True, backward=True, limit=1000, white_list=None) -> list:
        res = []
        if forward:
            r = self.forward_relations.get(mid, None)  # cache
            if r is not None:  # got from cache
                res += r
            else:  # query
                sparql = SparqlGenerator("?p", ns(mid), "?p", "?o").english_filter("?o").predicate_filters(
                    "?p").add_p_filter_list(p_filter_list, '?p').add_o_filter(o_filter, '?o').set_limit(
                    limit).to_sparql()
                query_res = self.query(sparql)
                if query_res is not None and len(query_res) != 0:
                    r = []
                    for ans in query_res:
                        r.append(ans['p'])
                    self.forward_relations[mid] = r
                    self.update_count['forward_relations'] = True
                    res += r
        if backward:
            r = self.backward_relations.get(mid, None)  # cache
            if r is not None:
                res += r
            else:  # query
                sparql = SparqlGenerator("?p", "?s", "?p", ns(mid)).english_filter("?s").predicate_filters(
                    "?p").add_p_filter_list(p_filter_list, '?p').add_s_filter(s_filter, '?s').set_limit(limit).to_sparql()
                query_res = self.query(sparql)
                if query_res is not None and len(query_res) != 0:
                    r = []
                    for ans in query_res:
                        r.append(ans['p'])
                    self.backward_relations[mid] = r
                    self.update_count['backward_relations'] = True
                    res += r

        if white_list is not None:
            res_white = []
            for r in res:
                if isinstance(r, dict):
                    r = r['value']
                if remove_ns(r) in white_list or r in white_list:
                    res_white.append(r)
            res = res_white
        return res

    def relation_value_by_mid_list(self, mid_list: list, forward=True, backward=True, white_list=None, with_cvt=False, sampling=False, prefix=True):
        if mid_list is not None and type(mid_list) != list:
            mid_list = [mid_list]
        assert type(mid_list) == list
        res = []
        for mid in mid_list:
            if is_mid_gid(mid) is False:  # not a mid
                continue
            mid_relation_list = self.relation_by_mid(mid, forward=forward, backward=backward, white_list=white_list)  # directly connected relations
            if not with_cvt:  # only directly connected relations
                res += get_values_in_list(mid_relation_list)
            elif with_cvt:  # including CVTs
                mid_relation_set = set()  # relation set for this mid
                for r1 in mid_relation_list:
                    if isinstance(r1, dict):
                        r1 = r1['value']
                    nei_list = self.neighbor_value_by_mid_and_relation(ns(mid), r1, forward=forward, backward=backward)  # mid, r1, nei1
                    for nei in nei_list:
                        if not self.is_cvt_node(nei):  # if this neighbor is not CVT
                            mid_relation_set.add(r1)
                        else:  # if this neighbor is CVT
                            nei1_relation_list = self.relation_value_by_mid_list([nei], forward=forward, backward=backward, white_list=white_list, with_cvt=False)
                            for r2 in nei1_relation_list:  # nei1, r2, nei2
                                if backward and not forward:
                                    t = (r2, r1)  # the order of CVT predicates
                                else:
                                    t = (r1, r2)
                                mid_relation_set.add(t)
                        if sampling:
                            break
                mid_relation_list = list(mid_relation_set)
                res += mid_relation_list
        if prefix is False:
            for i in range(0, len(res)):
                res[i] = remove_ns(res[i])
        return res

    def cvt_relation_value_by_mid_list(self, mid_list: list, relation_to_cvt: str, forward=True, backward=True,
                                       white_list=None):
        relation_list = []
        for mid in mid_list:
            if forward:
                neighbors = self.neighbor_value_by_mid_and_relation(mid, ns(relation_to_cvt), forward=True, backward=False)
                for nei in neighbors:  # for each CVT neighbors
                    if not self.is_cvt_node(nei):
                        continue
                    relation_list += self.relation_value_by_mid_list([nei], forward=forward, backward=backward, white_list=white_list)
            if backward:
                neighbors = self.neighbor_value_by_mid_and_relation(mid, ns(relation_to_cvt), forward=False, backward=True)
                for nei in neighbors:  # for each CVT neighbors
                    if not self.is_cvt_node(nei):
                        continue
                    relation_list += self.relation_value_by_mid_list([nei], forward=forward, backward=backward, white_list=white_list)
        return relation_list

    def neighbor_by_mid(self, mid: str, s_filter=None, o_filter=None, forward=True, backward=True) -> list:
        res = []
        if forward:
            sparql = SparqlGenerator("?o", ns(mid), "?p", "?o").english_filter("?o").predicate_filters("?p").add_o_filter(o_filter, '?o').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(ans['o'])
        if backward:
            sparql = SparqlGenerator("?s", "?s", "?p", ns(mid)).english_filter("?s").predicate_filters("?p").add_s_filter(s_filter, '?s').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(ans['s'])
        return res

    def neighbor_by_mid_list(self, mid_list, forward=True, backward=True):
        if type(mid_list) == str:
            mid_list = [mid_list]
        res = []
        for mid in mid_list:
            res += self.neighbor_by_mid(mid, forward=forward, backward=backward)
        return list(set(res))

    def neighbor_by_mid_and_relation(self, mid: str, relation_uri: str, s_filter=None, o_filter=None, forward=True, backward=True) -> list:
        res = []
        if is_mid_gid(mid) is False:
            return res
        if forward:
            sparql = SparqlGenerator('?o', ns(mid), relation_uri, '?o').english_filter('?o').add_o_filter(o_filter, '?o').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(ans['o'])
        if backward:
            sparql = SparqlGenerator('?s', '?s', relation_uri, ns(mid)).english_filter('?s').add_s_filter(s_filter, '?s').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(ans['s'])
        return res

    def neighbor_value_by_mid_and_relation(self, mid: str, relation: str, s_filter=None, o_filter=None, forward=True, backward=True) -> list:
        """
        Args:
            mid: mid
            relation: URI of Freebase relation
        Returns:
            a list of neighbors
        """
        neighbors = []
        if not is_mid_gid(mid):  # not mid, no neighbor
            return neighbors
        if forward:
            if remove_ns(mid) in self.forward_neighbors:  # cache
                nei = self.forward_neighbors[remove_ns(mid)].get(remove_ns(relation), [])
                if len(nei) != 0:
                    neighbors += nei
                else:  # query
                    try:
                        neighbors += self.neighbor_by_mid_and_relation(mid, relation, s_filter, o_filter, True, False)
                        self.forward_neighbors[remove_ns(mid)][remove_ns(relation)] = neighbors
                        self.update_count['forward_neighbors'] = True
                    except Exception as e:
                        print(e)
            else:  # query
                try:
                    neighbors += self.neighbor_by_mid_and_relation(mid, relation, s_filter, o_filter, True, False)
                    self.forward_neighbors[remove_ns(mid)] = dict()
                    self.forward_neighbors[remove_ns(mid)][remove_ns(relation)] = neighbors
                    self.update_count['forward_neighbors'] = True
                except Exception as e:
                    print(e)

        if backward:
            if remove_ns(mid) in self.backward_neighbors:  # cache
                nei = self.backward_neighbors[remove_ns(mid)].get(remove_ns(relation), [])
                if len(nei) != 0:
                    neighbors += nei
                else:  # query
                    try:
                        neighbors += self.neighbor_by_mid_and_relation(mid, relation, s_filter, o_filter, False, True)
                        self.backward_neighbors[remove_ns(mid)][remove_ns(relation)] = neighbors
                        self.update_count['backward_neighbors'] = True
                    except Exception as e:
                        print(e)
            else:  # query
                try:
                    neighbors += self.neighbor_by_mid_and_relation(mid, relation, s_filter, o_filter, False, True)
                    self.backward_neighbors[remove_ns(mid)] = dict()
                    self.backward_neighbors[remove_ns(mid)][remove_ns(relation)] = neighbors
                    self.update_count['backward_neighbors'] = True
                except Exception as e:
                    print(e)

        res = []
        for item in neighbors:
            if isinstance(item, dict):
                item = item['value']
            res.append(item)
        return res

    def triple_by_mid(self, mid: str, s_filter=None, o_filter=None, forward=True, backward=True, no_literal=False) -> list:
        """
        Get one-hop triples by a mid on Freebase
        Args:
            mid: a machine identifier on Freebase
            s_filter: entity filter for subject
            o_filter: entity filter for object
            forward: whether search the forward direction
            backward: whether search the backward direction
        Returns:
            A triple list.
        """
        res = []
        if forward:
            sparql = SparqlGenerator("?p ?o", ns(mid), "?p", "?o").english_filter("?o").predicate_filters("?p").add_o_filter(o_filter, '?o').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(Triple(ns(mid), ans['p']['value'], ans['o']['value'], s_type="uri", p_type=ans['p']['type'], o_type=ans['o']['type']))
        if backward:
            sparql = SparqlGenerator("?s ?p", "?s", "?p", ns(mid)).english_filter("?s").predicate_filters("?p").add_s_filter(s_filter, '?s').to_sparql()
            query_res = self.query(sparql)
            for ans in query_res:
                res.append(Triple(ans['s']['value'], ans['p']['value'], ns(mid), s_type=ans['s']['type'], p_type=ans['p']['type'], o_type="uri"))

        if no_literal:
            new_res = []
            for t in res:
                if t.s_type == 'uri' and t.o_type == 'uri':
                    new_res.append(t)
            res = new_res
        return res

    def triple_by_relation(self, relation: str, no_literal=False) -> list:
        res = []  # triple list
        sparql = SparqlGenerator('?s ?o', '?s', ns(relation), '?o').to_sparql()
        query_res = self.query(sparql)
        for ans in query_res:
            res.append(Triple(ans['s']['value'], ns(relation), ans['o']['value'], s_type=ans['s']['type'], o_type=ans['o']['type']))

        if no_literal:
            new_res = []
            for t in res:
                if t.s_type == 'uri' and t.o_type == 'uri':
                    new_res.append(t)
            res = new_res
        return res

    def triple_by_relation_list(self, relation_list: list, no_literal=False, distinct=True):
        res = []
        if type(relation_list) == str:
            relation_list = [relation_list]
        count = 0
        for r in relation_list:
            res += self.triple_by_relation(r, no_literal=no_literal)
            count += 1
            if count % 1000 == 0:
                print("get triples by relation: {}/{}".format(count, len(relation_list)))
        if distinct:
            res = list(set(res))
        return res

    def triple_by_mid_list(self, mid_list: list, s_filter=None, o_filter=None, forward=True, backward=True, no_literal=False, distinct=True) -> list:
        """
        Get one-hop triples by a mid list.
        Args:
            mid_list: a list of Freebase mid.
            s_filter: filtered subject
            o_filter: filtered object
            forward: whether the results contains forward triples.
            backward: whether the results contains backward triples.
        Returns:
            A list of one-hop triples for the mid list.
        """

        res = []
        if type(mid_list) == str:
            mid_list = [mid_list]
        count = 0
        for mid in mid_list:
            res += self.triple_by_mid(mid, s_filter, o_filter, forward, backward, no_literal=no_literal)  # triple list for this mid
            count += 1
            if count % 1000 == 0:
                print("get triples by mid: {}/{}".format(count, len(mid_list)))
        if distinct:
            res = list(set(res))
        return res

    def triple_and_entity_by_relation_list(self, relation_list):
        triples = set()
        entities = set()
        if type(relation_list) == str:
            relation_list = [relation_list]
        for idx in range(0, len(relation_list)):
            t = self.triple_by_relation(relation_list[idx])
            triples.update(t)
            entities.update([triple[0] for triple in t])
            entities.update([triple[2] for triple in t])

            if idx % 100 == 0:
                print("{}/{}, #entity: {}, #triple: {}".format(idx, len(relation_list), len(entities), len(triples)))
        return triples, entities

    def entity_name_by_mid(self, mid: str) -> list:
        mid = remove_ns(mid)
        if is_mid_gid(mid) is False:  # return the input directly
            return [mid]
        return self.entity_name_by_uri(mid)

    def entity_name_by_uri(self, uri: str) -> list:
        res = self.names.get(uri, None)
        if res is not None:
            return res
        try:
            sparql = SparqlGenerator('?o', ns(uri), 'http://rdf.freebase.com/ns/type.object.name', '?o').to_sparql()
            query_res = self.query_var(sparql, 'o')
            if query_res is None or len(query_res) == 0:
                self.names[uri] = [uri]  # return the input directly
            else:
                self.names[uri] = query_res
            self.update_count['name'] = True
            return self.names[uri]
        except Exception as e:
            print(e)
        return [uri]

    def entity_alias_by_mid(self, mid: str) -> str:
        if is_mid_gid(mid) is False:
            return [mid]
        res = self.alias.get(remove_ns(mid), None)
        if res is not None:  # got from cache
            return res
        sparql = SparqlGenerator("?o", ns(mid), 'http://rdf.freebase.com/ns/common.topic.alias', "?o").to_sparql()
        query_res = self.query_var(sparql, 'o')
        if query_res is None or len(query_res) == 0:
            return [mid]
        self.alias[mid] = query_res  # update cache
        self.update_count['alias'] = True
        return query_res

    def entity_name_and_alias_by_mid(self, mid: str):
        res = []
        res += self.entity_name_by_mid(mid)
        res += self.entity_alias_by_mid(mid)
        return res

    def entity_name_list_by_mid_list(self, mid_list):
        res = []
        if type(mid_list) == str:
            mid_list = [mid_list]
        for mid in mid_list:
            res += (self.entity_name_by_mid(mid))
        return res

    def entity_name_and_alias_by_mid_list(self, mid_list):
        res = []
        if type(mid_list) == str:
            mid_list = [mid_list]
        for mid in mid_list:
            res.append((mid, self.entity_name_and_alias_by_mid(mid)))
        return res

    def entity_key_by_mid(self, mid: str) -> list:
        """
        Get type.object.key of an entity by mid
        Args:
            mid: entity mid
        Returns:
            entity key
        """
        sparql = SparqlGenerator('?o', ns(mid), 'http://rdf.freebase.com/ns/type.object.key', '?o').to_sparql()
        query_res = self.query(sparql)
        res = []
        for ans in query_res:
            res.append(ans['o'])
        return res

    def rdf_label_by_mid(self, mid: str) -> list:
        sparql = SparqlGenerator('?o', ns(mid), 'http://www.w3.org/2000/01/rdf-schema#label', '?o').to_sparql()
        query_res = self.query(sparql)
        res = []
        for ans in query_res:
            res.append(ans['o']['value'])
        return res

    def rdf_type_by_uri(self, uri: str, common_filter=False) -> list:
        if self.rdf_type_cache is not None and remove_ns(uri) in self.rdf_type_cache:
            res = self.rdf_type_cache[remove_ns(uri)]
        else:
            sparql = SparqlGenerator('?o', ns(uri), 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', '?o').to_sparql()
            query_res = self.query(sparql)
            res = []
            for ans in query_res:
                res.append(ans['o']['value'])
            if self.rdf_type_cache is not None:
                self.rdf_type_cache[remove_ns(uri)] = res
                self.update_count['rdf_type'] = True

        if common_filter is True:
            new_res = []
            for rdf_type in res:
                if 'ns/common.' in rdf_type or 'ns/base.' in rdf_type or 'ns/freebase.' in rdf_type or 'rdf-schema' in rdf_type:
                    continue
                new_res.append(rdf_type)
            res = new_res
        return res

    def description_by_mid(self, mid: str) -> str:
        sparql = SparqlGenerator("?o", ns(mid), "http://rdf.freebase.com/ns/common.topic.description", "?o").english_filter("?o").to_sparql()
        query_res = self.query(sparql)
        res = []
        for ans in query_res:
            res.append(ans['o'])
        return res

    def triple_by_mid_and_relation(self, mid: str, relation_uri: str, s_filter=None, o_filter=None,
                                   forward=True, backward=True) -> list:
        res_list = []
        if forward:
            sparql = SparqlGenerator('?o', ns(mid), relation_uri, '?o').add_o_filter(o_filter).english_filter('?o').to_sparql()
            query_res = self.query_var(sparql, 'o')
            for o in query_res:
                res_list.append(Triple(ns(mid), relation_uri, o))

        if backward:
            sparql = SparqlGenerator('?s', '?s', relation_uri, ns(mid)).add_s_filter(s_filter).english_filter('?s').to_sparql()
            query_res = self.query_var(sparql, 's')
            for s in query_res:
                res_list.append(Triple(s, relation_uri, ns(mid)))
        return res_list

    def triple_by_mid_and_relation_list(self, mid: str, relation_list: list, s_filter=None, o_filter=None, forward=True, backward=True) -> list:
        res_list = []
        for relation in relation_list:
            res_list += self.triple_by_mid_and_relation(mid, relation, s_filter, o_filter, forward, backward)
        return res_list

    def is_entity_pair_in_k_hops(self, mid1: str, mid2: str, k=2):
        if k >= 1:
            sparql1 = SparqlGenerator('?p', mid1, '?p', mid2).english_filter('?p').to_sparql()
            sparql2 = SparqlGenerator('?p', mid2, '?p', mid1).english_filter('?p').to_sparql()
            query_res1 = self.query_var(sparql1, 'p')
            query_res2 = self.query_var(sparql2, 'p')
            if len(query_res1) != 0 or len(query_res2) != 0:
                return True

        if k >= 2:
            sparql1 = SparqlGenerator('?p1 ?p2', mid1, '?p1', '?o').add_triple('?o', '?p2', mid2).english_filter('?p1').english_filter('?p2').to_sparql()
            sparql2 = SparqlGenerator('?p1 ?p2', mid2, '?p1', '?o').add_triple('?o', '?p2', mid1).english_filter('?p1').english_filter('?p2').to_sparql()
            query_res1 = self.query_var(sparql1, 'p1')
            query_res2 = self.query_var(sparql2, 'p1')
            if len(query_res1) != 0 or len(query_res2) != 0:
                return True
        return False

    def triples_between_entity_pair(self, mid1: str, mid2: str):
        sparql1 = SparqlGenerator('?p', ns(mid1), '?p', ns(mid2)).english_filter('?p').to_sparql()
        sparql2 = SparqlGenerator('?p', ns(mid2), '?p', ns(mid1)).english_filter('?p').to_sparql()
        query_res1 = self.query_var(sparql1, 'p')
        query_res2 = self.query_var(sparql2, 'p')
        if len(query_res1):
            return query_res1
        if len(query_res2):
            return query_res2
        sparql1 = SparqlGenerator('?p1 ?p2', ns(mid1), '?p1', '?o').add_triple('?o', '?p2', ns(mid2)).english_filter('?p1').english_filter('?p2').to_sparql()
        sparql2 = SparqlGenerator('?p1 ?p2', ns(mid2), '?p1', '?o').add_triple('?o', '?p2', ns(mid1)).english_filter('?p1').english_filter('?p2').to_sparql()
        p1 = self.query_var(sparql1, 'p1')
        p2 = self.query_var(sparql1, 'p2')
        if len(p1) + len(p2):
            return p1 + p2
        p1 = self.query_var(sparql2, 'p1')
        p2 = self.query_var(sparql2, 'p2')
        if len(p1) + len(p2):
            return p1 + p2
        return []

    def triples_between_entity_and_literal(self, mid: str, literal: str):
        sparql = SparqlGenerator('?p', ns(mid), '?p', add_quotation_mark(literal)).english_filter('?p').to_sparql()
        p = self.query_var(sparql, 'p')
        if len(p):
            return p

        sparql = SparqlGenerator('?p1 ?p2', ns(mid), '?p1', '?o').add_triple('?o', '?p2', add_quotation_mark(literal)).english_filter('?p1').english_filter('?p2').to_sparql()
        p1 = self.query_var(sparql, 'p1')
        p2 = self.query_var(sparql, 'p2')
        if len(p1) + len(p2):
            return p1 + p2
        return []

    def is_cvt_node(self, mid: str) -> bool:
        """
        Determine whether a node is a CVT node or not in Freebase
        Args:
            mid: the node mid
        Returns:
            True if the node is a CVT node, False otherwise
        """
        if mid is None:
            return False
        if not ('m.' in remove_ns(mid) or 'g.' in remove_ns(mid)):  # not a mid or gid, thus not a CVT
            return False
        entity_name = self.entity_name_by_mid(ns(mid))
        if entity_name is None or len(entity_name) == 0 or entity_name[0] == mid or entity_name[0] == ns(mid):
            # no rdf:label or the label is just the mid
            return True
        return False

    def predicate_domain_and_range(self, predicate: str):
        predicate = ns(predicate)
        return self.predicate_domain(predicate), self.predicate_range(predicate)

    def predicate_domain(self, predicate: str, form='friendly_name'):
        if form == 'friendly_name':
            return predicate.split('.')[-2].replace('_', ' ')
        elif form == 'uri_no_prefix':
            idx = predicate.rfind('.')
            if idx != -1:
                return remove_ns(predicate[:idx])
            else:
                return remove_ns(predicate)
        return ''

    def predicate_range(self, predicate: str, form='friendly_name'):
        rn = remove_ns(predicate)
        if rn in self.predicate_range_cache:
            if form == 'friendly_name':
                return self.predicate_range_cache[rn].split('.')[-1].replace('_', ' ')
            elif form == 'uri_no_prefix':
                return self.predicate_range_cache[rn]
        try:
            sparql = SparqlGenerator('?x', ns(predicate), 'http://www.w3.org/2000/01/rdf-schema#range', '?x').to_sparql()
            p_range = self.query_var(sparql, 'x')
            if len(p_range) != 0:
                return p_range[0].split('.')[-1].replace('_', ' ')
        except Exception as e:
            print('predicate range: ' + str(e))
            return None
        return ''

    def predicate_list_concatenation(self, predicate_list):
        """
        Generate concatenated predicates.
        Args:
            predicate_list: a list of predicates, CVT maybe connected.
        Returns:
            the string of the concatenated predicates.
        """
        res = ''
        if type(predicate_list) == str:
            predicate_list = [predicate_list]

        i = 0
        for p in predicate_list:
            p = remove_ns(p)
            p_split = p.split('.')[-2:]
            if len(p_split) > 1:
                t = p_split[0].replace('_', ' ') + ', ' + p_split[1].replace('_', ' ')  # domain, label;
            else:
                t = p

            if i != len(predicate_list) - 1:  # not the last one
                res += (t + '; ')
            else:  # the last one
                p_range = self.predicate_range(p.replace('ns:', ''))
                if len(p_range) != 0:
                    res += (t + ', ' + p_range)
                else:
                    res += t
            i += 1
        return res

    def entity_name_and_mid_tuple_by_mid(self, mid: str):
        name = self.entity_name_by_mid(mid)
        return name, mid

    def entity_name_and_mid_tuple_list_by_mid_list(self, mid_list: list):
        if type(mid_list) == str:
            mid_list = [mid_list]

        mid_set = set(mid_list)
        res = []
        for mid in mid_set:
            res.append(self.entity_name_and_mid_tuple_by_mid(mid))
        return res

    def time_predicates_by_mid(self, mid: str, white_list=None):
        res = set()
        relations = self.relation_by_mid(mid, forward=True, backward=False, white_list=white_list)
        for r in relations:
            if r['type'] == 'uri' and r['value'].startswith('http://rdf.freebase.com/ns/'):
                if r['value'].endswith('from') or r['value'].endswith('to'):
                    res.add(r['value'])
        return res

    def time_predicates_by_mid_list(self, mid_list: list, white_list=None):
        if type(mid_list) == str:
            mid_list = [mid_list]

        res = set()
        for mid in mid_list:
            res = res.union(self.time_predicates_by_mid(mid, white_list=white_list))
        return list(res)

    def existential_mid_list_in_sparql(self, sparql_query: str):
        res = []
        if 'FILTER (?x != ?c)' not in sparql_query:
            return res
        sparql_query = sparql_query.replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ?c')
        return self.query_var(sparql_query, '?c')

    def instance_by_uri(self, uri):
        sparql = SparqlGenerator('?o', ns(uri), freebase_instance_uri, '?o').to_sparql()
        query_res = self.query(sparql)
        res = []
        for ans in query_res:
            res.append(ans['o']['value'])
        return res

    def reverse_relation(self, relation: str):
        assert self.reverse_relation_cache is not None
        if relation in self.reverse_relation_cache:
            return self.reverse_relation_cache[relation]
        return None

    def reverse_relation_list(self, relations):
        res = set()
        for r in relations:
            if r in self.reverse_relation_cache:
                res.add(self.reverse_relation_cache[r])
        return list(res)

    def get_property_by_domain(self, domain: str, prefix=False):
        res = set()
        domain = remove_ns(domain)
        for key in fb_domain_dict:
            if key not in domain:
                continue
            for prop in fb_domain_dict[key]:
                if prop.startswith(domain) and prop != domain:
                    res.add(prop)
        res = list(res)
        if prefix:  # return with prefix
            for i in range(0, len(res)):
                res[i] = ns(res[i])
        return res

    def get_property_by_range(self, rang: str, prefix=False):
        res = set()
        rang = remove_ns(rang)
        for t in self.roles:
            if len(t) >= 3 and t[2] == rang:
                res.add(t[1])
        res = list(res)
        if prefix:  # return with prefix
            for i in range(0, len(res)):
                res[i] = ns(res[i])
        return res

    def save_cache(self):
        if 'names' in self.update_count:
            pickle_save(self.names, self.cache_dir + '/names.bin')
        if 'forward_relations' in self.update_count:
            pickle_save(self.forward_relations, self.cache_dir + '/forward.bin')
        if 'backward_relations' in self.update_count:
            pickle_save(self.backward_relations, self.cache_dir + '/backward.bin')
        if 'alias' in self.update_count:
            pickle_save(self.alias, self.cache_dir + '/alias.bin')
        if 'forward_neighbors' in self.update_count:
            pickle_save(self.forward_neighbors, self.cache_dir + '/forward_nei.bin')
        if 'backward_neighbors' in self.update_count:
            pickle_save(self.backward_neighbors, self.cache_dir + '/backward_nei.bin')
        if 'rdf_type' in self.update_count:
            pickle_save(self.rdf_type_cache, self.cache_dir + '/rdf_type.bin')


if __name__ == "__main__":
    retriever = FreebaseRetriever(freebase_cache_dir)
