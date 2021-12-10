# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List

logger = logging.getLogger(__name__)


class RegisterWorld:
    WebQSP = 'WebQSP'
    GrailQA = 'GrailQA'

    @classmethod
    def values(cls):
        return {attr[1] for attr in vars(cls).items()
                if not callable(getattr(cls, attr[0])) and not attr[0].startswith("__")}


class GeneralizeLevel:
    iid = 1
    com = 2
    zero = 3


class EntityEncode:
    # Example: CLASS: measurement_unit.volume_unit ENTITY: cubic mile
    # [CLS] cubic mile [SEP]
    self = 'self'
    # [CLS] measurement unit volume unit cubic mile [SEP]
    all_domain = 'all_domain'
    # [CLS] measurement unit [unused1] volume unit [unused1] cubic mile [SEP]
    sep_all_domain = 'sep_all_domain'
    # [CLS] measurement unit cubic mile [SEP]
    top_domain = 'top_domain'
    # [CLS] measurement unit [unused1] cubic mile [SEP]
    sep_top_domain = 'sep_top_domain'
    # [CLS] volume unit cubic mile [SEP]
    close_domain = 'close_domain'
    # [CLS] volume unit [unused1] cubic mile [SEP]
    sep_close_domain = 'sep_close_domain'

    @classmethod
    def values(cls):
        return {attr[1] for attr in vars(cls).items()
                if not callable(getattr(cls, attr[0])) and not attr[0].startswith("__")}


class Universe:
    entity_repr = 'ENT'
    class_repr = 'CLASS'
    relation_repr = 'REL'


class JoinPredicate:
    entity_repr = 'JOIN_ENT'
    relation_repr = 'JOIN_REL'
    class_repr = 'JOIN_CLASS'


class EntityType:
    entity_str = "entity"
    entity_set = "class"
    entity_num = "literal"


class ComputableEntityClass:
    type_int = "type.int"
    type_float = "type.float"
    type_datetime = "type.datetime"
    type_year = "type.year"
    type_month = "type.month"
    type_date = "type.date"

    @classmethod
    def values(cls):
        return {attr[1] for attr in vars(cls).items()
                if not callable(getattr(cls, attr[0])) and not attr[0].startswith("__")}


class Entity:
    """
    Representing entity
    """

    def __init__(self,
                 # m.0pm2fgf
                 entity_id: str,
                 # opera.opera_production
                 entity_class: str,
                 # The Telephone / The Medium
                 friendly_name: str,
                 # entity
                 node_type: str,
                 # for building graph
                 node_id: int = None,
                 all_entity_classes: List[str] = None):
        # private variable
        self._node_id = node_id
        # entity id for indexing entity in Freebase
        self.entity_id = entity_id
        # may contain some unused, separated by space
        self.text = friendly_name
        # entity, class, literal
        self.entity_type = node_type
        assert self.entity_type in [EntityType.entity_str,
                                    EntityType.entity_set,
                                    EntityType.entity_num]
        # text: 1st domain. 2nd domain. 3rd domain ...
        # literal: type.boolean, type.datetime
        # other literals are stored in NumberEntity
        self.entity_class = entity_class

        if all_entity_classes is not None:
            self.all_entity_classes = all_entity_classes
        else:
            self.all_entity_classes = [entity_class]

    def __str__(self):
        return f'Ent:{self.entity_type}:{self.text}'

    def __hash__(self):
        return hash(f'Ent:{self.entity_type}:{self.text}')

    def __eq__(self, other):
        if isinstance(other, Class) and self.entity_id == other.entity_id:
            return True
        else:
            return False


class Class:
    """
    Representing entity
    """

    def __init__(self,
                 # m.0pm2fgf
                 entity_id: str,
                 # opera.opera_production
                 entity_class: str,
                 # The Telephone / The Medium
                 friendly_name: str,
                 # entity
                 node_type: str,
                 # for building graph
                 node_id: int = None,
                 all_entity_classes: List[str] = None):
        # private variable
        self._node_id = node_id
        # entity id for indexing entity in Freebase
        self.entity_id = entity_id
        # may contain some unused, separated by space
        self.text = friendly_name
        # entity, class, literal
        self.entity_type = node_type
        assert self.entity_type in [EntityType.entity_str,
                                    EntityType.entity_set,
                                    EntityType.entity_num]
        # text: 1st domain. 2nd domain. 3rd domain ...
        # literal: type.boolean, type.datetime
        # other literals are stored in NumberEntity
        self.entity_class = entity_class

        if all_entity_classes is not None:
            self.all_entity_classes = all_entity_classes
        else:
            self.all_entity_classes = [entity_class]

    def __str__(self):
        return f'Ent:{self.entity_class}:{self.text}'

    def __hash__(self):
        return hash(f'Ent:{self.entity_class}:{self.text}')

    def __eq__(self, other):
        if isinstance(other, Class) and self.entity_id == other.entity_id:
            return True
        else:
            return False


class NumberEntity(Entity):

    def __init__(self,
                 # special id for number
                 # as `num_index` in the question to represent it
                 entity_id: str,
                 friendly_name: str,
                 entity_class: str,
                 entity_value: str,
                 entity_index: int,
                 all_entity_classes: List[str] = None):
        super(NumberEntity, self).__init__(
            entity_id=entity_id,
            entity_class=entity_class,
            friendly_name=friendly_name,
            node_type="literal",
            node_id=None,
            all_entity_classes=all_entity_classes
        )

        # value is used for executing commands
        # and so it is not str type
        self.value = entity_value

        # entity_index is for relating this entity
        # with text on specific positions in question
        self.index = entity_index

    def __str__(self):
        return f'Num:{self.entity_class}:{self.text}'

    @classmethod
    def entity_to_number(cls, entity_id, entity_class, friendly_name, all_entity_classes):
        return NumberEntity(
            entity_id=entity_id,
            friendly_name=friendly_name,
            entity_class=entity_class,
            entity_value=friendly_name,
            entity_index=0,
            all_entity_classes=all_entity_classes
        )


class Relation:
    """
    Representing relation
    """

    def __init__(self,
                 # opera.opera_designer_gig.design_role
                 relation_class: str,
                 # Design Role
                 friendly_name: str,
                 # for building graph
                 node_start: int = None,
                 node_end: int = None):
        self._node_start = node_start
        self._node_end = node_end
        # 1st domain. 2nd domain. 3rd domain ...
        self.relation_id = relation_class
        self.relation_class = relation_class
        # used to represent the relation
        self.text = friendly_name

    def __str__(self):
        return f'Rel:{self.relation_class}'

    def __hash__(self):
        return hash(f'Rel:{self.relation_class}')

    def __eq__(self, other):
        if isinstance(other, Relation) and self.relation_class == other.relation_class:
            return True
        else:
            return False


UNK_CLASS = Class(entity_id="unk_class",
                  entity_class="unk",
                  friendly_name="You can not see me here",
                  node_type=EntityType.entity_set)

UNK_ENT = Entity(entity_id="unk_ent",
                 entity_class="unk",
                 friendly_name="You can not see me here",
                 node_type=EntityType.entity_str)

UNK_REL = Relation(relation_class="unk",
                   friendly_name="You can not see me here")


def extract_sxpression_structure(sexpression: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = sexpression.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def structure_to_flatten_string(sexpression: List) -> str:
    def flatten_sub_expression(sub_expression: List):
        prefix = "(" + sub_expression[0]
        for sub in sub_expression[1:]:
            if isinstance(sub, list):
                prefix += " " + flatten_sub_expression(sub)
            else:
                prefix += " " + sub
        prefix += ")"
        return prefix

    return flatten_sub_expression(sexpression)


def recursive_traverse_structure(struc_expression: List, keyword: str, results: List) -> List:
    """
    :param struc_expression: input structured expression as ['count', ['division', 'first']]
    :param keyword: the predict name of clauses you want to collect
    :param results: store the final structures which hit the keyword
    :return: note the list in results are also inter-connected, which means that if you modify
    results[0], it will affect results[1], which is in our expect.
    """
    # the first is function name
    assert len(struc_expression) > 1
    if struc_expression[0] == keyword:
        results.append(struc_expression)
    for sub in struc_expression[1:]:
        if isinstance(sub, list):
            recursive_traverse_structure(sub, keyword, results)
