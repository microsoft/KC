# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pickle
import sys
import traceback
from typing import Callable, Dict, List, Tuple, Set

from languages import SExpressionLanguage, WebQSPSExpressionLanguage
from sparql_executor import exec_verify_sparql_xsl_only_fix_literal
from utils import ComputableEntityClass, Universe
from utils import Relation, Entity, Class
from worlds.grail_world import KBWorld
from functools import reduce
from copy import copy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EARLY_STOPPING = "EARLY STOPPING HERE"
ANY_STRING = "ANY"
MAXIMUM_PARAMETER_LEN = 1000


class TreeNode(object):

    def __init__(self, node_slot, node_id, node_val, child=[]):
        """
        Build a tree node
        :param node_slot: value is either AND_OP, or arguments for inference
        :param child: for leaf node, child_nodes is set as None; otherwise, it is allocated as place-holder.
        """
        self.node_id = node_id
        self.node_val = node_val
        # the slot is used to judge if satisfy the requirement
        # slot values may by altered frequently and slot values of
        # intermediate nodes only depend on the leaf nodes
        self.node_slot = node_slot
        self.requirement = None
        self.child = child

    def able_to_reduce(self, node_table) -> bool:
        """
        test if an action could be inserted into self's child, if fail, return false; otherwise, return true.
        :return:
        """
        # if is a non terminal
        if None in [node_table[ind].node_val for ind in self.child]:
            return False
        # successfully add the child, return true.
        return all([node_table[ind].able_to_reduce(node_table) for ind in self.child])

    def add_child(self, action_node, node_table):
        ind = [node_table[ind].node_val for ind in self.child].index(None)
        # keep the original requirement
        action_node.requirement = node_table[self.child[ind]].requirement
        # keep the original reference
        self.child[ind] = action_node.node_id

    def update_slot(self, slot_val):
        self.node_slot = slot_val

    def export_sexpression(self, node_table):
        # export current node into standard sexpression
        if len(self.child) == 0:
            # TODO: we do not use slot since it will be changed frequently
            return self.node_val
        else:
            lchild_slot = node_table[self.child[0]].export_sexpression(node_table)
            if len(self.child) == 1:
                return "({} {})".format(self.node_val, lchild_slot)
            else:
                rchild_slot = node_table[self.child[1]].export_sexpression(node_table)
                return "({} {} {})".format(self.node_val, lchild_slot, rchild_slot)


def satisfy_require(requirement: Set, value: Set):
    """
    whether condition satisfies the requirement, e.g. there is any one of value inside requirement
    :param requirement: List[Optional[Tuple, str]]
    :param value: List[Optional[Tuple, str]]
    :return: if match, return True; otherwise, return False.
    """

    def set_compatiable(_val: Set, _require: Set):
        if ANY_STRING in _require:
            return True
        if ANY_STRING in _val:
            return True
        return any(_val & _require)

    if requirement is None or len(requirement) == 0:
        # no requirement
        return True
    else:
        assert isinstance(requirement, set)
        assert isinstance(value, set)
        # no valid value pass
        if len(value) == 0:
            return False
        elif type(next(iter(value))) != type(next(iter(requirement))):
            return False
        elif any(requirement & value):
            return True
        # require_ins and cond_ins may be `.+`
        if isinstance(next(iter(value)), str):
            # to speed up
            if ANY_STRING in requirement or ANY_STRING in value:
                return True
        else:
            # at least one position of entity is ANY_STRING
            require_zero, require_one = zip(*requirement)
            value_zero, value_one = zip(*value)
            require_zero, require_one, value_zero, value_one = set(require_zero), set(require_one), \
                                                               set(value_zero), set(value_one)
            return set_compatiable(value_zero, require_zero) & set_compatiable(value_one, require_one)
        return False


def _infer_reverse(entity_tuples: Set[Tuple]) -> Set[Tuple]:
    # given the relation as parameter, return its reverse
    return {(class_tuple[1], class_tuple[0]) for class_tuple in entity_tuples}


def _validate_join_ent(entity_tuples: Set[Tuple]) -> Set[str]:
    return {example[1] for example in entity_tuples}


def _infer_join_ent(entity_tuples: Set[Tuple], entities: Set[str]) -> Set[str]:
    # inner join which returns entities
    results = {example[0] for example in entity_tuples
               if example[1] in entities or example[1] == ANY_STRING}
    if len(results) > MAXIMUM_PARAMETER_LEN:
        return {ANY_STRING}
    else:
        return results


def _validate_join_rel(entity_tuples: Set[Tuple]) -> Set[Tuple]:
    return {(class_tuple[1], ANY_STRING) for class_tuple in entity_tuples}


def _infer_join_rel(entity_tuples1: Set[Tuple], entity_tuples2: Set[Tuple]) -> Set[Tuple]:
    # entity_tuples1: b1
    # entity_tuples2: b2
    # Inner join based on the first element of items in b2 and the second element of items in b1
    entity_tuples1_entities = {example[1] for example in entity_tuples1}
    entity_tuples2_entities = {example[0] for example in entity_tuples2}
    join_entities = entity_tuples1_entities & entity_tuples2_entities
    # A -> ANY, ANY -> B ---> A x B
    if ANY_STRING in entity_tuples1_entities or ANY_STRING in entity_tuples2_entities:
        join_entities.add(ANY_STRING)

    infer_entity_tuples = set()
    for out_example in entity_tuples1:
        # if ANY in entity 2, we should not skip
        if out_example[1] not in join_entities and ANY_STRING not in entity_tuples2_entities:
            continue
        for in_example in entity_tuples2:
            # if ANY, we should add all
            if in_example[0] not in join_entities and out_example[1] != ANY_STRING:
                continue
            infer_entity_tuples.add((out_example[0], in_example[1]))
    if (ANY_STRING, ANY_STRING) in infer_entity_tuples or len(infer_entity_tuples) > MAXIMUM_PARAMETER_LEN:
        infer_entity_tuples = {(ANY_STRING, ANY_STRING)}
    return infer_entity_tuples


def _validate_math(entity_tuples: Set[Tuple]) -> Set[str]:
    """
    Given an relation, its second parameter must be computable.
    """
    if any([example[1] for example in entity_tuples
            if example[1] in ComputableEntityClass.values() or example[1] == ANY_STRING]):
        return ComputableEntityClass.values()
    else:
        return {EARLY_STOPPING}


def _infer_math(entity_tuples: Set[Tuple], numbers: Set[str]):
    # entity_tuples: (x, v)
    # numbers: n
    # Return all x such that v inside (x,v) < / ≤ / > / ≥ n
    return {example[0] for example in entity_tuples}


def _validate_and_op(entities: Set[str]) -> Set[str]:
    # at least the second argument should contain current entity class
    return entities


def _infer_and_op(entities1: Set[str], entities2: Set[str]) -> Set[str]:
    if ANY_STRING in entities1 or ANY_STRING in entities2:
        return {ANY_STRING}
    else:
        return entities1 & entities2


def _infer_count(entities: Set[str]) -> Set[str]:
    # TODO: the return value of COUNT should be compatible with NumberEntity
    #  no requirement on input entity class
    return {ComputableEntityClass.type_int}


def _validate_arg(entities: Set[str]) -> Set[Tuple]:
    """
    Constraint on its neighbor relation: return List[Tuple]
    """
    # you should generate a requirement for relation - entity class tuples
    return {(example, ANY_STRING) for example in entities}


def _infer_arg(entities: Set[str], entity_tuples: Set[Tuple]) -> Set[str]:
    # you should generate a requirement for relation - entity class tuples
    return entities


def _validate_tc_now(entities: Set[str]) -> Set[Tuple]:
    return {(ANY_STRING, ANY_STRING)}


def _infer_tc_now(entities: Set[str], entity_tuples: Set[Tuple]) -> Set[str]:
    return {ANY_STRING}


def _infer_tc(entities: Set[str], entity_tuples: Set[Tuple], number_entity: int) -> Set[str]:
    return {ANY_STRING}


class SExpressionState:
    action_validate_dict = {
        SExpressionLanguage.AND_OP.__name__: _validate_and_op,
        SExpressionLanguage.JOIN_ENT.__name__: _validate_join_ent,
        SExpressionLanguage.JOIN_REL.__name__: _validate_join_rel,
        SExpressionLanguage.JOIN_CLASS.__name__: _validate_join_ent,
        SExpressionLanguage.ARGMAX.__name__: _validate_arg,
        SExpressionLanguage.ARGMIN.__name__: _validate_arg,
        SExpressionLanguage.lt.__name__: _validate_math,
        SExpressionLanguage.le.__name__: _validate_math,
        SExpressionLanguage.gt.__name__: _validate_math,
        SExpressionLanguage.ge.__name__: _validate_math,
        # extra action name
        WebQSPSExpressionLanguage.TC_NOW.__name__: _validate_tc_now,
        WebQSPSExpressionLanguage.TC.__name__: _validate_tc_now
    }

    action_infer_dict = {
        SExpressionLanguage.AND_OP.__name__: _infer_and_op,
        SExpressionLanguage.R.__name__: _infer_reverse,
        SExpressionLanguage.COUNT.__name__: _infer_count,
        SExpressionLanguage.JOIN_ENT.__name__: _infer_join_ent,
        SExpressionLanguage.JOIN_REL.__name__: _infer_join_rel,
        SExpressionLanguage.JOIN_CLASS.__name__: _infer_join_ent,
        SExpressionLanguage.ARGMAX.__name__: _infer_arg,
        SExpressionLanguage.ARGMIN.__name__: _infer_arg,
        SExpressionLanguage.lt.__name__: _infer_math,
        SExpressionLanguage.le.__name__: _infer_math,
        SExpressionLanguage.gt.__name__: _infer_math,
        SExpressionLanguage.ge.__name__: _infer_math,
        # extra action name
        WebQSPSExpressionLanguage.TC_NOW.__name__: _infer_tc_now,
        WebQSPSExpressionLanguage.TC.__name__: _infer_tc
    }

    def __init__(self,
                 possible_actions,
                 world: KBWorld,
                 is_nonterminal: Callable[[str], bool],
                 relation_to_argument: Dict[str, List] = None,
                 anchor_to_argument: Dict[str, Dict] = None,
                 literal_relation: Set[str] = None,
                 entity_to_argument: Dict[str, str] = None,
                 model_in_training: bool = False,
                 enabled_type: bool = False,
                 enabled_virtual: bool = False,
                 enabled_anchor: bool = False,
                 enabled_runtime_prune: bool = False):
        self.possible_actions = possible_actions
        # argument stack stores the requirement (e.g. .+, and specific ones)

        self.world = world
        self.language = world.language
        self.is_nonterminal = is_nonterminal
        self.enabled_virtual = enabled_virtual
        self.enabled_anchor = enabled_anchor
        self.enabled_runtime_prune = enabled_runtime_prune
        self.enabled_type = enabled_type

        # if in training, no checking is executed; otherwise, depends on each option.
        self.enabled = (not model_in_training) & (enabled_virtual | enabled_anchor |
                                                  enabled_runtime_prune | enabled_type)

        # FORMAT SPEC: relation to argument is as
        # "education.educational_institution.school_type": [
        #         {"education.educational_institution":
        #         "education.school_category"}
        #     ]
        # ]
        if entity_to_argument is None:
            self.entity_to_argument = {}
            for entity in self.world.kb_context.entity_list:
                # all arguments as list to be compatible
                if len(entity.all_entity_classes) > 0:
                    self.entity_to_argument[entity.entity_id] = set(entity.all_entity_classes)
                    if "type.type" in self.entity_to_argument[entity.entity_id]:
                        self.entity_to_argument[entity.entity_id].remove("type.type")
        else:
            self.entity_to_argument = entity_to_argument

        self.relation_to_argument = relation_to_argument
        self.anchor_to_argument = anchor_to_argument
        self.literal_relation = literal_relation

        if self.anchor_to_argument.keys():
            self.anchor_in_relations = reduce(lambda x, y: set(x) | set(y),
                                              [self.anchor_to_argument[key]['in_relation']
                                               for key in self.anchor_to_argument.keys()])
            self.anchor_out_relations = reduce(lambda x, y: set(x) | set(y),
                                               [self.anchor_to_argument[key]['out_relation']
                                                for key in self.anchor_to_argument.keys()])
        else:
            self.anchor_in_relations = {}
            self.anchor_out_relations = {}

        self.node_accessed = False
        # if this is set as True, remove all linked ones to construct a dead-loop one
        self.early_stop = False
        self.action_history = []

        # manually maintain the connection between different node
        self.node_table: Dict[int, TreeNode] = {}
        self.node_queue: List[int] = []

    def copy_self(self):
        # shallow copy
        new_sexpression_state = copy(self)

        # deep copy
        new_sexpression_state.action_history = self.action_history[:]
        new_sexpression_state.node_queue = self.node_queue[:]
        # only the node table record the necessary node information
        new_sexpression_state.node_table = pickle.loads(pickle.dumps(self.node_table))

        return new_sexpression_state

    def collapse_finished_nodes_and_validate(self, node_queue, node_table, validate=True) -> bool:
        while len(node_queue) > 0 and node_table[node_queue[-1]].able_to_reduce(node_table):
            cur_node = node_table[node_queue[-1]]
            # call function
            func_name = cur_node.node_slot
            # using
            parameters = [node_table[node_id].node_slot for node_id in cur_node.child]
            # inference on the entity class level
            infer_slot = self.action_infer_dict[func_name](*parameters)

            if validate:
                match_requirement = satisfy_require(cur_node.requirement, infer_slot)
                if not match_requirement:
                    # TODO: return None to avoid decoding on this sequence
                    return False

            # update slot
            cur_node.update_slot(infer_slot)
            # to its parent
            node_queue.pop(-1)
        return True

    def runtime_prune(self, node_queue, node_table) -> Tuple[bool, List[str]]:
        """
        Given current stack of nodes and node_tables, returns the largest node which can be reduced
        Note this function will not effect the node status, it just convert it into a sexpression
        """
        if not (len(node_queue) > 0 and node_table[node_queue[-1]].able_to_reduce(node_table)):
            return False, []
        # reduce from top to down
        try:
            history_nontermnial = set()
            existing_predicates = {"JOIN_REL", "JOIN_ENT", "JOIN_REL"}
            for i in range(len(node_queue)):
                # Only JOIN utilizes the KB information
                cur_node: TreeNode = node_table[node_queue[i]]
                history_nontermnial.add(cur_node.node_val)
                if cur_node.able_to_reduce(node_table):
                    if cur_node.node_val in existing_predicates:
                        s_expression = cur_node.export_sexpression(node_table)
                        # this s_expression should be post-processed to process JOIN_REL and etc.
                        s_expression = self.world.postprocess(s_expression)
                        query = self.world.sparql_converter(s_expression)
                        # obtain slot values List[List[str]]
                        slot_values, status_code = exec_verify_sparql_xsl_only_fix_literal(query)
                        flatten_slot_values = [slot[0] for slot in slot_values]
                        if status_code == 200:
                            return True, flatten_slot_values
                        else:
                            return False, []
        except Exception as e:
            print("Error on runtime_prune:\n")
            exec_info = sys.exc_info()
            traceback.print_exception(*exec_info)
            # no way to execute
            return True, []
        return False, []

    def detect_early_stop(self, history: List[str]):
        # to avoid invalid rules to rank the first
        lhs_seq = [rule.split(' -> ')[0] for rule in history]
        rhs_first_seq = [rule.split(' -> ')[1].strip().split(', ')[0] for rule in history]
        rhs_seq = [rule.split(' -> ')[1].strip().split(', ') for rule in history]
        # @start@ -> Class, Class -> instance
        if len(lhs_seq) == 2 and lhs_seq[1] == Class.__name__ and len(rhs_seq[1]) == 1:
            return True
        # <Class,Class:Class> -> AND_OP, Class -> ins1, Class -> ins2
        if SExpressionLanguage.AND_OP.__name__ in rhs_first_seq:
            for i in range(len(rhs_first_seq) - 2):
                if rhs_first_seq[i] == SExpressionLanguage.AND_OP.__name__ and \
                        len(rhs_seq[i + 1]) == 1 and len(rhs_seq[i + 2]) == 1:
                    return True
        # <Relation:Relation> -> R, Relation -> [<Relation:Relation>, Relation], <Relation:Relation> -> R
        if SExpressionLanguage.R.__name__ in rhs_first_seq:
            for i in range(len(rhs_first_seq) - 2):
                if rhs_first_seq[i] == SExpressionLanguage.R.__name__ and \
                        rhs_first_seq[i + 2] == SExpressionLanguage.R.__name__:
                    return True
        return False

    def take_action(self, production_rule: str) -> 'SExpressionState':
        if not self.enabled or self.early_stop:
            return self

        new_sql_state = self.copy_self()

        lhs, rhs = production_rule.split(' -> ')
        rhs_tokens = rhs.strip('[]').split(', ')

        # TODO: if you write the language by inheriting DomainLanguage,
        #  multiple tokens must not be an terminal production rule
        if len(rhs_tokens) == 1:
            is_terminal = not self.is_nonterminal(rhs)
            # use right side to build tree node
            if is_terminal:
                # AND_OP, JOIN_REL and etc.
                if self.world.language.is_global_rule(rhs):
                    # WARNING: this only applies for language which follows the DomainLanguage
                    last_action = new_sql_state.action_history[-1]
                    lhs, rhs_param = last_action.split(' -> ')
                    num_parameters = len(rhs_param.strip('[]').split(', ')) - 1

                    # Here we want to alloc a non-terminal node
                    # And so the node_id is actually node_ids
                    child_nodes = []
                    for _ in range(num_parameters):
                        node_id = len(new_sql_state.node_table)
                        child_node = TreeNode(node_slot=None, node_val=None, node_id=node_id)
                        new_sql_state.node_table[node_id] = child_node
                        child_nodes.append(node_id)

                    node_id = len(new_sql_state.node_table)
                    tree_node = TreeNode(node_slot=rhs,
                                         node_val=rhs,
                                         node_id=node_id,
                                         child=child_nodes)
                    new_sql_state.node_table[node_id] = tree_node
                # if relation, append entity tuples; otherwise, append entity class
                else:
                    if rhs in new_sql_state.entity_to_argument:
                        infer_slot = new_sql_state.entity_to_argument[rhs]
                    elif rhs in new_sql_state.relation_to_argument:
                        infer_slot = new_sql_state.relation_to_argument[rhs]
                    else:
                        # TODO: hard code for relations which cannot be found in freebase
                        if self.language.obtain_leaf_type(rhs) == Universe.entity_repr:
                            infer_slot = {ANY_STRING}
                        else:
                            infer_slot = {(ANY_STRING, ANY_STRING)}

                    # add new node to node table
                    node_id = len(new_sql_state.node_table)
                    tree_node = TreeNode(node_slot=infer_slot,
                                         node_val=rhs,
                                         node_id=node_id)
                    new_sql_state.node_table[node_id] = tree_node

                # assign to expression node
                if new_sql_state.node_accessed is False:
                    # set as True
                    new_sql_state.node_accessed = True
                else:
                    # collapse the full tree
                    new_sql_state.node_table[new_sql_state.node_queue[-1]]. \
                        add_child(tree_node, new_sql_state.node_table)

                if self.world.language.is_global_rule(rhs):
                    new_sql_state.node_queue.append(tree_node.node_id)

        # We use the nearest grammar rule (JOIN_ENT, AND and so on) and its
        # first argument to identify if the next parameter could satisfy its
        # type constraint. E.g. ( JOIN_REL ( R ( relation_1 ) entity_1 ) )
        # the first parameter of JOIN_REL is reverse of relation_1
        # When each grammar rule ends, it will be used to infer the type of its
        # next parameter (with the help of action history)
        new_sql_state.action_history.append(production_rule)
        early_stop = self.detect_early_stop(new_sql_state.action_history)
        if early_stop:
            new_sql_state.early_stop = True
        else:
            # must do this
            result = self.collapse_finished_nodes_and_validate(new_sql_state.node_queue,
                                                               new_sql_state.node_table,
                                                               validate=self.enabled_virtual)
            if result is False:
                new_sql_state.early_stop = True

        return new_sql_state

    def get_valid_actions(self, valid_actions: dict):
        if not self.enabled or self.early_stop or 'linked' not in valid_actions:
            return valid_actions

        valid_actions_ids = [rule_id for rule_id in valid_actions['linked'][2]]
        valid_actions_rules = [self.possible_actions[rule_id] for rule_id in valid_actions_ids]

        # use lhs to determine whether to trigger
        lhs_nonterminal = valid_actions_rules[0].split(' -> ')[0]

        last_requirement = None

        # execute operator on the argument if the minimum count is satisfied
        if len(self.node_queue) > 0:
            # take the nearest node in node queue
            peek_node = self.node_table[self.node_queue[-1]]
            if len(peek_node.child) > 1 and \
                    self.node_table[peek_node.child[0]].node_slot and \
                    self.node_table[peek_node.child[1]].requirement is None:
                # pop an operator and an argument, and execute it
                operator_name = peek_node.node_val
                first_argument = self.node_table[peek_node.child[0]].node_slot
                # TODO: note that all _validate functions return the minimum requirement
                #  not the strict requirement
                next_argument_require = self.action_validate_dict[operator_name](first_argument)
                # must be set
                next_argument_require = set(next_argument_require)
                # reduce the space
                if len(next_argument_require) >= MAXIMUM_PARAMETER_LEN:
                    if isinstance(next(iter(next_argument_require)), Tuple):
                        next_argument_require = {(ANY_STRING, ANY_STRING)}
                    else:
                        next_argument_require = {ANY_STRING}
                # append into argument stack for checking
                # WARNING: somehow may be hard code
                self.node_table[peek_node.child[1]].requirement = next_argument_require
                last_requirement = next_argument_require

        actions_to_remove = set()

        # instance-level checking
        if self.enabled_anchor:
            # 1. <Relation,Entity:Class> -> JOIN_ENT, (Optional) Relation -> [<Relation:Relation>, Relation],
            # (Optional) <Relation:Relation> -> R, Relation -> ?
            # proactive predicting on the valid relations based on CANDIDATE entities
            if lhs_nonterminal == Relation.__name__ and self.literal_relation:
                normal_setting = self.action_history[-1].split(' -> ')[1] == SExpressionLanguage.JOIN_ENT.__name__
                reverse_setting = self.action_history[-1].split(' -> ')[1] == SExpressionLanguage.R.__name__ and \
                                  self.action_history[-3].split(' -> ')[1] == SExpressionLanguage.JOIN_ENT.__name__
                if normal_setting or reverse_setting:
                    for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                        temp_rel_rhs = rule.split(' -> ')[1]
                        # literal relations cannot be for judged
                        if temp_rel_rhs in self.literal_relation:
                            continue
                        if normal_setting and temp_rel_rhs not in self.anchor_in_relations:
                            actions_to_remove.add(rule_id)
                        elif reverse_setting and temp_rel_rhs not in self.anchor_out_relations:
                            actions_to_remove.add(rule_id)
            # 2. <Relation,Entity:Class> -> JOIN_ENT, (Optional) Relation -> [<Relation:Relation>, Relation],
            # (Optional) <Relation:Relation> -> R, Relation -> rel, Entity -> ?
            # post-checking to prune invalid relations
            if lhs_nonterminal == Entity.__name__:
                action_rules = self.action_history[-1].split(' -> ')
                reverse_flag = self.action_history[-2].split(' -> ')[1] == SExpressionLanguage.R.__name__
                should_remove = action_rules[0] == Relation.__name__
                if should_remove:
                    for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                        temp_ent_rhs = rule.split(' -> ')[1]
                        if temp_ent_rhs in self.anchor_to_argument:
                            if reverse_flag:
                                check_relations = self.anchor_to_argument[temp_ent_rhs]['out_relation']
                            else:
                                check_relations = self.anchor_to_argument[temp_ent_rhs]['in_relation']
                            if action_rules[1] not in check_relations:
                                actions_to_remove.add(rule_id)
            # update ids and rules
            valid_actions_ids = [action_id for action_id in valid_actions_ids if action_id not in actions_to_remove]
            valid_actions_rules = [self.possible_actions[rule_id] for rule_id in valid_actions_ids]

        # type-level checking
        if self.enabled_type and last_requirement is not None:
            # preprocess to identify valid actions
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                _, rhs = rule.split(' -> ')
                if rhs in self.entity_to_argument:
                    # decision on an entity
                    # last_requirement: List[str]
                    # condition:  List[str]
                    condition = self.entity_to_argument[rhs]
                    match_requirement = satisfy_require(last_requirement, condition)
                elif rhs in self.relation_to_argument:
                    # decision on an relation
                    # last_requirement: List[Tuple[str, str]]
                    # condition:  List[Tuple[str, str]]
                    condition = self.relation_to_argument[rhs]
                    match_requirement = satisfy_require(last_requirement, condition)
                else:
                    match_requirement = True

                if not match_requirement:
                    actions_to_remove.add(rule_id)
            # update ids and rules
            valid_actions_ids = [action_id for action_id in valid_actions_ids if action_id not in actions_to_remove]
            valid_actions_rules = [self.possible_actions[rule_id] for rule_id in valid_actions_ids]

        # execution semantic level: virtual execution
        # try to apply each rule and decide if the filled one satisfy the requirement
        if self.enabled_virtual and len(self.node_queue) > 0:
            # virtually forward to try
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                _, rhs = rule.split(' -> ')
                if rhs in self.entity_to_argument:
                    # decision on an entity
                    # last_requirement: List[str]
                    # condition:  List[str]
                    argument = self.entity_to_argument[rhs]
                elif rhs in self.relation_to_argument:
                    # decision on an relation
                    # last_requirement: List[Tuple[str, str]]
                    # condition:  List[Tuple[str, str]]
                    argument = self.relation_to_argument[rhs]
                else:
                    # ignore these
                    continue

                # alloc a totally new node because we will change its child.
                # but for other nodes, we will only changed its slot
                # (which can be overwritten for many times)
                local_node_table = pickle.loads(pickle.dumps(self.node_table))
                local_node_queue = self.node_queue[:]

                # these nodes are temporal nodes, we should delete them after judgement
                temp_node_id = len(local_node_table)
                tree_node = TreeNode(node_slot=argument, node_id=temp_node_id, node_val=rhs)
                local_node_table[temp_node_id] = tree_node

                local_node_table[local_node_queue[-1]].add_child(tree_node, local_node_table)
                # the reference is changed after copying!
                result = self.collapse_finished_nodes_and_validate(local_node_queue,
                                                                   local_node_table)

                if result is False:
                    actions_to_remove.add(rule_id)
            # update ids and rules
            valid_actions_ids = [action_id for action_id in valid_actions_ids if action_id not in actions_to_remove]
            valid_actions_rules = [self.possible_actions[rule_id] for rule_id in valid_actions_ids]

        # execution semantic level: subprogram induction
        if self.enabled_runtime_prune and len(self.node_queue) > 0 and \
                len(valid_actions_ids) > 0 and \
                lhs_nonterminal in [Entity.__name__,
                                    Class.__name__,
                                    Relation.__name__]:
            # here we will fake a node and append it into current node table
            local_node_queue = self.node_queue[:]
            local_node_table = pickle.loads(pickle.dumps(self.node_table))
            # allocate node id and record it
            node_id = len(local_node_table)
            fake_tree_node = TreeNode(node_slot="SLOT", node_id=node_id, node_val="SLOT")
            local_node_table[node_id] = fake_tree_node
            local_node_table[local_node_queue[-1]].add_child(fake_tree_node, local_node_table)

            is_pruning, accept_slots = self.runtime_prune(node_queue=local_node_queue, node_table=local_node_table)
            if is_pruning:
                for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                    _, rhs = rule.split(' -> ')
                    if self.language.is_number_entity(rhs):
                        continue
                    if rhs not in accept_slots:
                        actions_to_remove.add(rule_id)

        # here we want to filter those ones which cannot be covered
        new_valid_actions = {}

        if 'global' in valid_actions:
            new_valid_actions['global'] = valid_actions['global']

        new_linked_actions = self._remove_actions(valid_actions, 'linked',
                                                  actions_to_remove) if 'linked' in valid_actions else None
        if new_linked_actions is not None:
            new_valid_actions['linked'] = new_linked_actions
        elif 'global' not in valid_actions:
            self.early_stop = True
            new_valid_actions['linked'] = valid_actions['linked']

        return new_valid_actions

    @staticmethod
    def _remove_actions(valid_actions, key, ids_to_remove):
        if len(ids_to_remove) == 0:
            return valid_actions[key]

        if len(ids_to_remove) == len(valid_actions[key][2]):
            return None

        current_ids = valid_actions[key][2]
        keep_ids = []
        keep_ids_loc = []

        for loc, rule_id in enumerate(current_ids):
            if rule_id not in ids_to_remove:
                keep_ids.append(rule_id)
                keep_ids_loc.append(loc)

        items = list(valid_actions[key])
        items[0] = items[0][keep_ids_loc]
        items[1] = items[1][keep_ids_loc]
        items[2] = keep_ids

        if len(items) >= 4:
            items[3] = items[3][keep_ids_loc]
        return tuple(items)
