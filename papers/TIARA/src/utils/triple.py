# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.uri_util import dbp_dbo_conversion, brackets_triple, dbp_prefix, dbo_prefix

TYPE_URI = 'uri'
TYPE_LITERAL = 'literal'
TYPE_VAR = 'var'


class Triple:
    s: str  # mid for freebase, dbo/dbr for DBpedia
    p: str
    o: str
    s_type: str
    p_type: str
    r_type: str
    weight: float  # optional
    directed: bool  # default: True

    def __init__(self, s: str, p: str, o: str, weight=None, s_type=None, p_type=None, o_type=None, directed=True):
        self.s = s
        self.p = p
        self.o = o
        self.weight = weight
        self.s_type = s_type
        self.p_type = p_type
        self.o_type = o_type
        if s_type is None:
            self.s_type = get_type(s)
        if p_type is None:
            self.p_type = get_type(p)
        if o_type is None:
            self.o_type = get_type(o)
        self.directed = directed

    def __str__(self) -> str:
        if self.directed is True:
            return '{' + self.s + ', ' + self.p + ', ' + self.o + '}'
        else:
            return '{' + self.s + ', ' + self.p + ', ' + self.o + '} UNION {' + self.o + ', ' + self.p + ', ' + self.s + '}'

    def to_sparql(self, dbp_dbo_union=True) -> str:
        if not (self.p.startswith(dbp_prefix) or self.p.startswith(dbo_prefix)):
            dbp_dbo_union = False

        if self.directed:
            res = brackets_triple(self.s, self.p, self.o)
            if dbp_dbo_union is True:
                res = '{' + res + '} UNION {' + brackets_triple(self.s, dbp_dbo_conversion(self.p), self.o) + '}'
        else:  # not directed
            res = '{ ' + brackets_triple(self.s, self.p, self.o) + '} UNION { ' + brackets_triple(self.o, self.p,
                                                                                                  self.s) + '}'
            if dbp_dbo_union is True:
                res = '{' + res + '} UNION {' + brackets_triple(self.s, dbp_dbo_conversion(self.p), self.o) \
                      + '} UNION { ' + brackets_triple(self.o, dbp_dbo_conversion(self.p), self.s) + '}'
        res += '. \n'
        return res

    def set_s(self, s):
        self.s = s
        self.s_type = get_type(s)

    def set_p(self, p):
        self.p = p
        self.p_type = get_type(p)

    def set_o(self, o):
        self.o = o
        self.o_type = get_type(o)

    def get_var_set(self):
        res = set()
        if self.s.startswith('?'):
            res.add(self.s)
        if self.p.startswith('?'):
            res.add(self.p)
        if self.o.startswith('?'):
            res.add(self.o)
        return res


def triple_list_to_str(triple_list: list) -> str:
    res = ''
    for triple in triple_list:
        res += (str(triple) + '\n')
    return res


def get_triple_weight(triple):
    return triple.weight


def get_type(s: str):
    if s.startswith('http://'):
        return TYPE_URI
    if s.startswith('?'):
        return TYPE_VAR
    return TYPE_LITERAL
