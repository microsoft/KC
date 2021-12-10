# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Callable
import numpy as np
from allennlp.common.checks import ConfigurationError
from kb_utils.kb_context import KBContext
from utils import EntityType, Universe


class Action:

    def __init__(self, action_str: str):
        self.repr = action_str
        struc_splits = action_str.split(" -> ")
        # assert to avoid -> appears in realtion/entity
        assert len(struc_splits) == 2
        self.nonterminal = struc_splits[0]
        # there may be many expanded nonterminals in rhs
        self.rhs = struc_splits[1]

    def __repr__(self):
        return self.repr


class KBWorld:
    """
    The world class is responsible for specifying all valid candidate actions for a specific case,
    and converting them into their corresponding index in model training.
    """

    def __init__(self, world_id: str, kb_context: KBContext, sparql_query: str, sexpression: str,
                 origin_utterance: str, answers: List[str], level: str, graph_query_info: Dict,
                 entity_meta_info: Dict, entity_offset_tokens: List[str],
                 language_class: Callable, sparql_converter: Callable, verify_sparql_converter: Callable):
        self.world_id = world_id
        self.kb_context = kb_context
        self.language = language_class.build(kb_context)

        # build parsed sexpression for evaluation
        self.sexpression_eval = sexpression
        # we pre-process it for parsing
        self.sexpression_parse = self._preprocess(sexpression)
        self.sparql_query = sparql_query

        self.origin_utterance = origin_utterance
        self.answers = answers
        self.level = level

        self.entities_indexer = {}
        for i, schema in enumerate(self.kb_context.knowledge_graph.entities):
            main_type = schema[:schema.find(":")]
            if main_type == Universe.entity_repr:
                # entity id is at the last, we should avoid it to be split
                parts = schema.split(":", maxsplit=3)
            elif main_type == Universe.relation_repr:
                parts = schema.split(":", maxsplit=2)
            else:
                raise ConfigurationError(f"Do not support for main type as {main_type}")
            self.entities_indexer[parts[-1]] = i

        self.valid_actions: Dict[str, List[str]] = {}
        self.valid_actions_flat: List[Action] = []
        self.graph_query_info = graph_query_info
        self.entity_meta_info = entity_meta_info
        self.entity_offset_tokens = entity_offset_tokens

        # a converter to convert sexpression to sparql
        self.sparql_converter = sparql_converter
        self.verify_sparql_converter = verify_sparql_converter

    def get_action_sequence_and_all_actions(self):
        try:
            action_sequence = self.language.logical_form_to_action_sequence(self.sexpression_parse)
        except:
            action_sequence = []

        all_action = self.language.all_possible_productions()

        # parse str into structure inside Action
        self.valid_actions_flat = [Action(ins) for ins in all_action]

        # build nested structure
        for action in self.valid_actions_flat:
            action_key = action.nonterminal
            if action_key not in self.valid_actions:
                self.valid_actions[action_key] = []
            # record action
            self.valid_actions[action_key].append(action.repr)

        return action_sequence, all_action

    def index_entity_type(self):
        defined_types = ['@@PAD@@', EntityType.entity_num, EntityType.entity_set,
                         EntityType.entity_str, Universe.relation_repr]

        # now we have 5 types
        assert len(defined_types) == 5

        # record the entity index
        entity_type_indices = []

        for entity_index, schema in enumerate(self.kb_context.knowledge_graph.entities):
            parts = schema.split(':')
            entity_main_type = parts[0]
            if entity_main_type == Universe.relation_repr:
                entity_type = defined_types.index(entity_main_type)
            elif entity_main_type == Universe.entity_repr:
                entity_coarse_type = parts[1]
                entity_type = defined_types.index(entity_coarse_type)
            else:
                raise ConfigurationError("Get the unknown entity: {}".format(schema))
            entity_type_indices.append(entity_type)

        return np.array(entity_type_indices, dtype=np.int)

    def get_action_entity_mapping(self) -> Dict[str, int]:
        """
        Get the entity index of every local grammar(also named after linked action)
        :return:
        """
        mapping = {}

        for action in self.valid_actions_flat:
            # default is padding
            mapping[str(action)] = -1

            # lowercase for all entities
            production_right = action.rhs

            # only instance class should apply entity map
            if production_right not in self.entities_indexer:
                continue

            # record the entity id
            mapping[str(action)] = self.entities_indexer[production_right]

        return mapping

    """
    Private functions to override
    """
    def _preprocess(self, sexpression: str):
        """
        1. processing all functions
        2. distinguish JOIN_ENT and JOIN_REL
        """
        return sexpression

    def postprocess(self, sexpression: str) -> str:
        """
        The reverse function for `preprocess`.
        """
        return sexpression
