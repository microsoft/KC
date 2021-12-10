# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sparql_executor import exec_sparql_fix_literal
from allennlp.common.testing import AllenNlpTestCase


class TestSparqlExecution(AllenNlpTestCase):

    def test_sparql_literal_result_1(self):
        exec_res = exec_sparql_fix_literal(
            "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.03wc52)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.03wc52 ns:tennis.tennis_player.year_turned_pro ?x .\n}\n")
        assert exec_res[0][0] == "2005"

    def test_sparql_literal_result_2(self):
        exec_res = exec_sparql_fix_literal(
            "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.083qr)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.083qr ns:people.person.spouse_s ?y .\n?y ns:people.marriage.from ?x .\n?y ns:people.marriage.type_of_union ns:m.04ztj .\n}\n")
        assert exec_res[0][0] == "1672-04-04"

    def test_sparql_literal_query(self):
        exec_res = exec_sparql_fix_literal(
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE { \nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :cvg.computer_videogame . \nVALUES ?x1 { \"2002-07-08:00\"^^<http://www.w3.org/2001/XMLSchema#gYearMonth> } \n?x0 :cvg.computer_videogame.release_date ?x1 . \nFILTER ( ?x0 != ?x1  ) \n} \n}")
        assert set(exec_res[0]) == {"m.07pwgx", "m.0yvxql3"}
