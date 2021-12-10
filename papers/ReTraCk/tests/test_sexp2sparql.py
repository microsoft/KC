# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sparql_executor import exec_sparql_fix_literal
from allennlp.common.testing import AllenNlpTestCase
from converter.converter_grailqa import lisp_to_sparql
from converter.converter_webqsp import webqsp_lisp_to_sparql

class TestSexp2SparqlExecution(AllenNlpTestCase):

    def test_grailqa_date_case1(self):
        query = lisp_to_sparql("(AND cvg.computer_videogame (JOIN cvg.computer_videogame.release_date 2002-07^^http://www.w3.org/2001/XMLSchema#gYearMonth))")
        result = exec_sparql_fix_literal(query)
        print(result)
        assert set(result[0]) == {"m.07pwgx", "m.0yvxql3"}

    def test_webqsp_date_case1(self):
        query = webqsp_lisp_to_sparql("(AND cvg.computer_videogame (JOIN cvg.computer_videogame.release_date 2002-07^^http://www.w3.org/2001/XMLSchema#gYearMonth))")
        result = exec_sparql_fix_literal(query)
        print(result)
        assert set(result[0]) == {"m.07pwgx", "m.0yvxql3"}

