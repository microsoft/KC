# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from languages import SExpressionLanguage
from grail_context import _get_numbers_from_tokens
from utils import extract_sxpression_structure, recursive_traverse_structure, Universe, JoinPredicate
from server import render_logic_form_template


class TestDomainLanguage(AllenNlpTestCase):

    def setup_method(self):
        super().setup_method()
        self.language = SExpressionLanguage(
            constant_ent={
                "m.0fq59pm": "Cyrano",
                "m.04l1gwb": "unkown thing",
                "m.01663r": "m.01663r",
                "m.03q44xr": "m.03q44xr"
            },
            constant_rel={
                "cvg.computer_game_engine_family.engines_in_this_family": "Engines in This Family",
                "amusement_parks.ride.opened": "Opened",
                "amusement_parks.ride_type.rides": "Rides",
                "amusement_parks.ride.capacity": "Capacity (riders per hour)",
                "people.person.profession": "people.person.profession",
                "people.profession.people_with_this_profession": "people.profession.people_with_this_profession"
            },
            constant_num={
                "2400^^http://www.w3.org/2001/XMLSchema#integer": "2400"
            },
            constant_cls={
                "amusement_parks.disney_ride": "Disney Ride",
                "amusement_parks.ride": "Amusement Ride",
                "amusement_parks.ride_type": "Amusement Ride Type",
                "government.politician": "government.politician",
                "cvg.computer_game_engine_family": "Video Game Engine Family"
            }
        )

    def test_simple_logical_form_1(self):
        action_seq = self.language.logical_form_to_action_sequence("(AND_OP cvg.computer_game_engine_family (JOIN_ENT cvg.computer_game_engine_family.engines_in_this_family m.0fq59pm))")
        print(action_seq)

    def test_simple_logical_form_2(self):
        action_seq = self.language.logical_form_to_action_sequence("(ARGMAX amusement_parks.disney_ride amusement_parks.ride.opened)")
        print(action_seq)

    def test_complex_logical_form(self):
        action_seq = self.language.logical_form_to_action_sequence("(AND_OP government.politician (AND_OP (JOIN_ENT (R people.profession.people_with_this_profession) m.04l1gwb) (JOIN_ENT people.person.profession m.01663r)))")
        print(action_seq)

    def test_number_logic_form(self):
        action_seq = self.language.logical_form_to_action_sequence("(AND_OP amusement_parks.ride (JOIN_ENT (R amusement_parks.ride_type.rides) (JOIN_ENT amusement_parks.ride_type.rides (lt amusement_parks.ride.capacity 2400^^http://www.w3.org/2001/XMLSchema#integer))))")
        print(action_seq)

    def test_all_valid_logic_form(self):
        all_valid_actions = self.language.all_possible_productions()
        print(all_valid_actions)

    def test_extract_text_number(self):
        sentence_with_number = "My son is 1800 years old"
        sentence_tokens = [Token(text=text) for text in sentence_with_number.split(" ")]
        number_list = _get_numbers_from_tokens(sentence_tokens)
        assert len(number_list) == 1
        assert number_list[0][0] == '1800'
        assert number_list[0][1] == 3

    def test_extract_float_number(self):
        sentence_with_number = "My son is 12.345 years old"
        sentence_tokens = [Token(text=text) for text in sentence_with_number.split(" ")]
        number_list = _get_numbers_from_tokens(sentence_tokens)
        assert len(number_list) == 1
        assert number_list[0][0] == '12.345'
        assert number_list[0][1] == 3

    def test_extract_empty_date(self):
        sentence_with_number = "My son is born in July or 2020-01-20"
        sentence_tokens = [Token(text=text) for text in sentence_with_number.split(" ")]
        number_list = _get_numbers_from_tokens(sentence_tokens)
        assert len(number_list) == 0

    def test_joint_preprocess(self):
        sexpression = "(AND digicams.camera_color_filter_array_type (JOIN (R digicams.digital_camera.color_filter_array_type) (JOIN (R digicams.camera_compressed_format.cameras) m.03q44xr)))"
        expression_struc = extract_sxpression_structure(sexpression)
        results = []
        recursive_traverse_structure(expression_struc, keyword="JOIN", results=results)
        # reverse results since small clauses are visited later
        results = list(reversed(results))
        for i, clause in enumerate(results):
            assert len(clause) == 3
            # only check for the second argument
            if isinstance(clause[2], list):
                semantic_type = self.language.obtain_join_type(clause[2][0])
            else:
                semantic_type = self.language.obtain_join_type(clause[2])
            # modify the JOIN name
            results[i][0] = semantic_type
        assert expression_struc[2][0] == JoinPredicate.class_repr
        assert expression_struc[2][2][0] == JoinPredicate.entity_repr

    def test_expression_into_str(self):
        def flatten_sub_expression(sub_expression: List):
            prefix = "(" + sub_expression[0]
            for sub in sub_expression[1:]:
                if isinstance(sub, list):
                    prefix += " " + flatten_sub_expression(sub)
                else:
                    prefix += " " + sub
            prefix += ")"
            return prefix

        sexpression = "(AND digicams.camera_color_filter_array_type (JOIN (R digicams.digital_camera.color_filter_array_type) (JOIN (R digicams.camera_compressed_format.cameras) m.03q44xr)))"
        expression_struc = extract_sxpression_structure(sexpression)
        expression_str = flatten_sub_expression(expression_struc)
        assert expression_str == sexpression

    def test_demo_html(self):
        sexpression = "(AND digicams.camera_color_filter_array_type (JOIN (R digicams.digital_camera.color_filter_array_type) (JOIN (R digicams.camera_compressed_format.cameras) m.03q44xr)))"
        render_html = render_logic_form_template(sexpression)
        print(render_html)