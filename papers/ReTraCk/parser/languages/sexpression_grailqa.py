# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict
from utils import Entity, Class, Relation, Universe, NumberEntity, EntityType, JoinPredicate

from allennlp_semparse.domain_languages.domain_language import (
    DomainLanguage,
    predicate,
)
from kb_utils.kb_context import KBContext

logger = logging.getLogger(__name__)


class SExpressionLanguage(DomainLanguage):

    def __init__(self, constant_ent: Dict[str, str],
                 constant_cls: Dict[str, str],
                 constant_rel: Dict[str, str],
                 # TODO: float or str?
                 constant_num: Dict[str, str]):
        super().__init__(start_types={
            Class,
            int
        })

        # to distinguish different JOIN predicates
        self.type_book = {}
        self.join_type_book = {}

        # record the id and its value for constants
        # TODO: consider date types
        # TODO: add type-specific predicates
        for key, value in constant_ent.items():
            self.add_constant(key, value, type_=Entity)
            self.type_book[key] = Universe.entity_repr
            # the second argument decides the JOIN predicate name
            self.join_type_book[key] = JoinPredicate.entity_repr

        for key, value in constant_num.items():
            self.add_constant(key, value, type_=NumberEntity)
            self.type_book[key] = Universe.entity_repr
            self.join_type_book[key] = JoinPredicate.entity_repr

        for key, value in constant_cls.items():
            self.add_constant(key, value, type_=Class)
            self.type_book[key] = Universe.class_repr
            self.join_type_book[key] = JoinPredicate.class_repr

        for key, value in constant_rel.items():
            self.add_constant(key, value, type_=Relation)
            self.type_book[key] = Universe.relation_repr
            self.join_type_book[key] = JoinPredicate.relation_repr

        assert JoinPredicate.entity_repr == SExpressionLanguage.JOIN_ENT.__name__
        assert JoinPredicate.class_repr == SExpressionLanguage.JOIN_CLASS.__name__
        assert JoinPredicate.relation_repr == SExpressionLanguage.JOIN_REL.__name__

    """
    Build from GrailKBContext
    """
    @classmethod
    def build(cls, kb_context: KBContext):
        constant_num = {number.entity_id: number.text for number in kb_context.number_list}
        constant_ent = {}
        constant_class = {}

        for i in reversed(range(len(kb_context.entity_list))):
            entity = kb_context.entity_list[i]
            if entity.entity_type in [EntityType.entity_str, EntityType.entity_num]:
                constant_ent[entity.entity_id] = entity.text
            elif entity.entity_type == EntityType.entity_set:
                constant_class[entity.entity_id] = entity.text

            if entity.entity_type == EntityType.entity_num:
                # do not remove from entity since they are entity
                constant_num[entity.entity_id] = entity.text

        constant_rel = {relation.relation_id: relation.text for relation in kb_context.relation_list}
        return cls(constant_ent, constant_class, constant_rel, constant_num)

    @predicate
    def AND_OP(self, entity_set_1: Class, entity_set_2: Class) -> Class:
        pass

    @predicate
    def R(self, relation_set: Relation) -> Relation:
        pass

    @predicate
    def COUNT(self, entity_set: Class) -> int:
        pass

    @predicate
    def JOIN_ENT(self, relation_set: Relation, entity: Entity) -> Class:
        pass

    @predicate
    def JOIN_REL(self, relation_set: Relation, second_relation_set: Relation) -> Relation:
        pass

    @predicate
    def JOIN_CLASS(self, relation_set: Relation, entity_set: Class) -> Class:
        pass

    @predicate
    def ARGMAX(self, entity_set: Class, relation_set: Relation) -> Class:
        pass

    @predicate
    def ARGMIN(self, entity_set: Class, relation_set: Relation) -> Class:
        pass

    @predicate
    def lt(self, relation_set: Relation, val: NumberEntity) -> Class:
        pass

    @predicate
    def le(self, relation_set: Relation, val: NumberEntity) -> Class:
        pass

    @predicate
    def gt(self, relation_set: Relation, val: NumberEntity) -> Class:
        pass

    @predicate
    def ge(self, relation_set: Relation, val: NumberEntity) -> Class:
        pass

    def obtain_leaf_type(self, predicate_name: str):
        """
        Given a predicate_name (or maybe a entity id or relation id), return its expected semantic type.
        This function is designed for distinguishing different JOIN predicates when preprocessing.
        :return: semantic type as Entity or Relation
        """
        if predicate_name in self.type_book:
            return self.type_book[predicate_name]
        else:
            return None

    def obtain_join_type(self, predicate_name: str):
        """
        Given a predicate_name (or maybe a entity id or relation id), return its expected semantic type.
        This function is designed for distinguishing different JOIN predicates when preprocessing.
        :return: semantic type as Entity or Relation
        """
        if predicate_name in self.type_book:
            return self.join_type_book[predicate_name]
        elif predicate_name in ["AND_OP", "JOIN_CLASS", "JOIN_ENT",
                                "ge", "gt", "le", "lt", "ARGMAX", "ARGMIN"]:
            return JoinPredicate.class_repr
        else:
            return JoinPredicate.relation_repr

    def is_global_rule(self, rhs_production: str):
        if rhs_production in self.type_book:
            return False
        else:
            return True

    def is_number_entity(self, entity_name):
        if entity_name not in self._function_types:
            return False
        elif NumberEntity.__name__ in [str(name) for name  in self._function_types[entity_name]]:
            return True
        return False
