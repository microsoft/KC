# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from SPARQLWrapper import SPARQLWrapper, JSON
from utils.uri_util import remove_question_mark, remove_ns


class KBRetriever:
    kb_name: str
    sparql_wrapper: SPARQLWrapper
    english_kb: bool

    def __init__(self, english_kb=True):
        self.english_kb = english_kb

    def query(self, sparql_query: str) -> list:
        self.sparql_wrapper.setQuery(sparql_query)
        self.sparql_wrapper.setReturnFormat(JSON)
        try:
            ret = self.sparql_wrapper.query().convert()
            # ret is a stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
            if 'results' in ret:
                bindings = ret['results']['bindings']
                res = []
                for bindings_item in bindings:
                    is_english = True
                    for var in bindings_item:
                        if self.english_kb and ('xml:lang' in bindings_item[var].keys()) and (
                                bindings_item[var]['xml:lang'] != 'en'):
                            is_english = False
                            break
                    # add bindings_item to res
                    if self.english_kb and is_english is False:
                        continue
                    res.append(bindings_item)
                return res
            elif 'boolean' in ret:
                return ret['boolean']
        except Exception as e:
            print(e)
            print('[ERROR] query exception:', sparql_query)
            return []

    def query_var(self, sparql_query: str, var) -> list:
        assert type(sparql_query) == str
        var = remove_question_mark(var)
        query_res = self.query(sparql_query)
        if type(query_res) == bool:  # ASK
            return query_res
        elif 'SELECT DISTINCT COUNT' in sparql_query or 'SELECT COUNT' in sparql_query:  # count
            var = 'callret-0'

        res = []
        if len(query_res) != 0:
            for ans in query_res:
                if len(ans) != 0:
                    if var in ans:
                        res.append(remove_ns(ans[var]['value']).replace('-08:00', ''))
                    else:
                        print('[WARN] the variable \'' + var + '\' is not in SPARQL: ' + sparql_query)
        return res

    def relation_value_by_mid_list(self, mid_list: list, forward=True, backward=True, white_list=None,
                                   with_cvt=False, sampling=False):
        raise NotImplementedError("Please Implement this method")

    def relations_of_var(self, sparal_query: str, var: str, forward=True, backward=False, white_list=None) -> list:
        res = []
        var = remove_question_mark(var)
        try:
            query_res = self.query_var(sparal_query, var)[:5]
            res = list(
                set(self.relation_value_by_mid_list(query_res, forward, backward, white_list, with_cvt=False)))
        except Exception as e:
            print('relations_of_var: ' + str(e))
        return res
