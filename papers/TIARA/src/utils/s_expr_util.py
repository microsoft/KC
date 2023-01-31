# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('retriever')

from retriever.executor.logic_form_util import lisp_to_sparql
from retriever.executor.sparql_executor import execute_query
from retriever.enumerate_candidates import get_approx_s_expr
from retriever.webqsp_eval_topk_prediction import get_time_macro_clause


def execute_s_expr(expr):
    if 'time_macro' in expr:
        try:
            approx_expr = get_approx_s_expr(expr)
        except:
            return 'null', []
        try:
            additional_clause = get_time_macro_clause(expr)
            approx_sparql = lisp_to_sparql(approx_expr)
            approx_sparql_end = approx_sparql.rfind('}')
            cat_sqarql = approx_sparql[:approx_sparql_end] + additional_clause + approx_sparql[approx_sparql_end:]

            cat_result = execute_query(cat_sqarql)

            return expr, cat_result
        except:
            return 'null', []
    else:
        # query_expr = expr.replace('( ', '(').replace(' )', ')')
        # return query_expr, []
        try:
            # print('parse', query_expr)
            sparql_query = lisp_to_sparql(expr)
            # print('sparql', sparql_query)
            denotation = execute_query(sparql_query)
        except:
            expr = 'null'
            denotation = []
        return expr, denotation
