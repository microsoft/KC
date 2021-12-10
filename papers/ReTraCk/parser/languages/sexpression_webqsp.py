# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from allennlp_semparse.domain_languages.domain_language import (
    predicate,
)
from overrides import overrides

from utils import Class, Relation, NumberEntity, JoinPredicate
from .sexpression_grailqa import SExpressionLanguage

logger = logging.getLogger(__name__)


class WebQSPSExpressionLanguage(SExpressionLanguage):

    @predicate
    def TC_NOW(self, entity_set: Class, relation: Relation) -> Class:
        pass

    @predicate
    def TC(self, entity_set: Class, relation: Relation, val: NumberEntity) -> Class:
        pass

    @overrides
    def obtain_join_type(self, predicate_name: str):
        """
        Given a predicate_name (or maybe a entity id or relation id), return its expected semantic type.
        This function is designed for distinguishing different JOIN predicates when preprocessing.
        :return: semantic type as Entity or Relation
        """
        if predicate_name in self.type_book:
            return self.join_type_book[predicate_name]
        elif predicate_name in ["AND_OP", "JOIN_CLASS", "JOIN_ENT",
                                "ge", "gt", "le", "lt", "ARGMAX", "ARGMIN", "TC_NOW", "TC"]:
            return JoinPredicate.class_repr
        else:
            return JoinPredicate.relation_repr
