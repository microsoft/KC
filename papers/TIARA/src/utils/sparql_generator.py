# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy


from utils.triple import Triple
from utils.uri_util import ns, brackets, add_question_mark


class SparqlGenerator:
    """
    The generator of SPARQL queries.
    """

    triple_list: list
    filter_list: list
    order_by: str
    lambda_var: str
    score: float
    limit: int
    offset: int
    distinct: bool
    query_type: str  # select, ask, count

    def __init__(self, lambda_var=None, s=None, p=None, o=None, order_by=None, limit=None, distinct=True, score=-1,
                 offset=None, query_type='select'):
        if lambda_var is not None:
            self.lambda_var = lambda_var
        elif query_type != 'ask':  # ans_var is None
            print('[ERROR] invalid SPARQL, set a lambda variable or set the query type to \'ask\'')
        self.triple_list = []
        self.filter_list = []

        if s is not None and p is not None and o is not None:
            self.triple_list.append(Triple(s, p, o))
        self.score = score
        self.order_by = order_by
        self.limit = limit
        self.offset = offset
        self.distinct = distinct
        self.query_type = query_type

    def add_triple(self, s, p, o, score=None, directed=True):
        self.triple_list.append(Triple(s, p, o, directed=directed))
        self.add_score(score)
        return self

    def add_filter(self, filter_str: str, score=None):
        if filter_str is not None:
            self.filter_list.append(filter_str)
        self.add_score(score)
        return self

    def add_order_by(self, var: str, sort: str, xsd=None, limit=1):
        var = add_question_mark(var)
        self.order_by = 'ORDER BY '
        if sort.lower() == 'desc':
            self.order_by += 'DESC '
        if xsd == 'datetime':
            var = 'xsd:datetime(' + var + ')'
        elif xsd == 'float':
            var = 'xsd:float(' + var + ')'
        elif xsd == 'int' or xsd == 'integer':
            var = 'xsd:integer(' + var + ')'
        self.order_by += '(' + var + ')'
        self.limit = limit
        return self

    def add_s_filter(self, s_filter: str, var=None):
        if s_filter is None:
            return self
        if var is None and len(self.triple_list) == 0:
            return self
        if var is None:
            var = self.triple_list[0].s
        self.filter_list.append(add_question_mark(var) + ' != ' + brackets(ns(s_filter)))
        return self

    def add_o_filter(self, o_filter: str, var=None):
        if o_filter is None:
            return self
        if var is None and len(self.triple_list) == 0:
            return self
        if var is None:
            var = self.triple_list[0].o
        self.filter_list.append(add_question_mark(var) + ' != ' + brackets(ns(o_filter)))
        return self

    def add_p_filter(self, p_filter: str, var: str):
        if p_filter is None:
            return self
        if var is None and len(self.triple_list) == 0:
            return self
        if var is None:
            var = self.triple_list[0].p
        self.filter_list.append('!regex(' + add_question_mark(var) + ',\'' + p_filter + '\',\'i\')')
        return self

    def add_p_filter_list(self, p_filter_list: list, var: str):
        if p_filter_list is None:
            return self
        for p_filter in p_filter_list:
            self.add_p_filter(p_filter, var)
        return self

    def extend(self, p: str, existential: str, score=0):
        """
        Extend the SPARQL query.
        Args:
            p: a new predicate
            existential: a new existential variable
            score: the score of this new predicate p

        Returns:
            the instance itself

        """
        for triple in self.triple_list:
            if triple.s == self.lambda_var:
                triple.set_s(existential)
            if triple.o == self.lambda_var:
                triple.set_o(existential)
        self.add_triple(existential, p, self.lambda_var)
        self.score += score
        return self

    def to_sparql(self) -> str:
        """
        Generate a SPARQL query
        Returns:
            a string of a SPARQL query
        """

        res = ''

        # Query type
        if self.query_type == 'select' or self.query_type == 'count':
            res = 'SELECT '
        elif self.query_type == 'ask':
            res = 'ASK WHERE '

        # DISTINCT
        if self.distinct and self.query_type != 'ask':
            res += 'DISTINCT '

        # lambda variable(s)
        if self.query_type == 'select':
            res += self.lambda_var + ' {\n'
        elif self.query_type == 'count':
            res += 'COUNT(' + self.lambda_var + ') {\n'
        elif self.query_type == 'ask':
            res += '{\n'

        # Filter
        for filter_str in self.filter_list:  # including entity / predicate filters
            res += ('FILTER (' + filter_str + ')\n')

        # Triple pattern
        for triple in self.triple_list:
            res += triple.to_sparql()
        res += '} '

        # Order by
        if self.order_by is not None and self.query_type != 'ask':
            res += self.order_by

        # Limit
        if self.limit is not None and self.limit >= 0:
            res += ' LIMIT ' + str(self.limit)
        return res

    def english_filter(self, var: str):
        """
        Add a English filter for an entity
        Args:
            var: a variable for entity

        Returns:
            the instance itself
        """
        if not var.strip(" ").startswith("?"):
            var = "?" + var
        self.filter_list.append(
            "!isLiteral(" + var + ") OR lang(" + var + ") = '' OR langMatches(lang(" + var + "), 'en')")
        return self

    def predicate_filters(self, var: str):
        if not var.strip(' ').startswith('?'):
            var = '?' + var
        self.filter_list.append(var + " != rdf:type")
        self.filter_list.append(var + " != rdfs:label")
        self.filter_list.append('!regex(' + var + ',\'common.topic.\',\'i\')')
        self.filter_list.append('!regex(' + var + ',\'wikipedia\',\'i\')')
        self.filter_list.append('!regex(' + var + ',\'type.object.key\',\'i\')')
        self.filter_list.append('!regex(' + var + ',\'common.topic.topic_equivalent_webpage\',\'i\')')
        self.filter_list.append('!regex(' + var + ',\'common.topic.webpage\',\'i\')')
        self.filter_list.append('!regex(' + var + ',\'user.avh\',\'i\')')
        self.filter_list.append(var + ' != <http://rdf.freebase.com/key/en>')
        self.filter_list.append(var + ' != <http://www.w3.org/2000/01/rdf-schema#range>')
        self.filter_list.append(var + ' != <http://www.w3.org/2000/01/rdf-schema#domain>')
        self.filter_list.append(var + ' != <http://www.w3.org/2002/07/owl#inverseOf>')
        return self

    def sample_predicate_filters(self, var: str):
        """
        Filters for sampling in the relation matching task.
        Args:
            var: a SPARQL var of a predicate

        Returns:
            the instance itself
        """
        self.predicate_filters(var)
        self.filter_list.append(var + ' != <http://rdf.freebase.com/ns/type.object.type>')
        self.filter_list.append(var + ' != <http://rdf.freebase.com/ns/type.object.name>')
        return self

    def set_limit(self, num_limit: int):
        if num_limit >= 0:
            self.limit = num_limit
        return self

    def set_distinct(self, d: bool):
        self.distinct = d
        return self

    def set_score(self, score: float):
        self.score = score
        return self

    def add_score(self, score: float):
        if score is not None:
            self.score += score
        return self

    def get_score(self):
        """
        Get the score of this SPARQL query.
        Returns:
            The SPARQL score.
        """
        if len(self.triple_list) == 0:
            return -1
        return self.score / len(self.triple_list)

    def get_var_set(self):
        res = set()
        for triple in self.triple_list:
            res = res.union(triple.get_var_set())
        return res

    def clone(self):
        """
        Get a copy of this SPARQL query.
        Returns:
            a clone of this SPARQL query.
        """
        return copy.deepcopy(self)

    def get_predicate_set(self):
        res = set()
        for triple in self.triple_list:
            res.add(triple.p)
        return res


def sparql_collection_to_str(sparql_generator_list):
    res = '[Predicted SPARQL] '
    if sparql_generator_list is None:
        return res + 'None'
    if type(sparql_generator_list) == SparqlGenerator:
        return res + sparql_generator_list.to_sparql() + '\n'
    for s in sparql_generator_list:
        if not isinstance(s, SparqlGenerator):
            continue
        res += s.to_sparql() + '\n'
    return res


def get_sparql_score(sparql_generator: SparqlGenerator):
    if len(sparql_generator.triple_list) == 0:
        return -1
    return sparql_generator.score / len(sparql_generator.triple_list)


def xsd_datetime_value(datetime: str):
    datetime = datetime.strip('\"')
    return '"' + datetime + '"^^xsd:dateTime'


def xsd_year_first_day(year):
    if isinstance(year, int):
        year = str(year)
    return '"' + year + '-01-01"^^xsd:dateTime'


def xsd_year_last_day(year):
    if isinstance(year, int):
        year = str(year)
    return '"' + year + '-12-31"^^xsd:dateTime'


def xsd_datetime_var(var: str):
    var = add_question_mark(var)
    return 'xsd:datetime(' + var + ')'


def get_triple_str(s, p, o):
    return s + ' ' + p + ' ' + o
