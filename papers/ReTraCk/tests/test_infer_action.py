# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from allennlp.common.testing import AllenNlpTestCase
from sexpression_state import _validate_and_op, _validate_arg, _validate_join_ent, _validate_join_rel, _validate_math
from sexpression_state import _infer_and_op, _infer_arg, _infer_count, _infer_join_ent, _infer_join_rel, _infer_math, _infer_reverse
from sexpression_state import satisfy_require


class TestDomainLanguage(AllenNlpTestCase):

    def setup_method(self):
        super().setup_method()
        self.ent_arg1 = {"type1", "type2", "type3"}
        self.rel_arg_1 = {("type1", "type2")}
        self.ent_any = {'ANY'}
        self.rel_any = {("ANY", "ANY")}

    def template(self, func1, func2, argument1):
        argument2 = func1(argument1)
        print(argument2)
        infer_value = func2(argument1, argument2)
        print(infer_value)

    def test_case_and(self):
        self.template(_validate_and_op,
                      _infer_and_op,
                      self.ent_arg1)
        self.template(_validate_math,
                      _infer_math,
                      self.rel_arg_1)
        self.template(_validate_join_rel,
                      _infer_join_rel,
                      self.rel_arg_1)
        self.template(_validate_join_ent,
                      _infer_join_ent,
                      self.rel_arg_1)
        self.template(_validate_arg,
                      _infer_arg,
                      self.ent_arg1)

    def test_case_any(self):
        self.template(_validate_and_op,
                      _infer_and_op,
                      self.ent_any)
        self.template(_validate_math,
                      _infer_math,
                      self.rel_any)
        self.template(_validate_join_rel,
                      _infer_join_rel,
                      self.rel_any)
        self.template(_validate_join_ent,
                      _infer_join_ent,
                      self.rel_any)
        self.template(_validate_arg,
                      _infer_arg,
                      self.ent_any)
