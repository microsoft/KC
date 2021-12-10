# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import traceback
from collections import namedtuple
from typing import Dict, List, Any, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Attention
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from torch.nn.modules import Dropout
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.training.metrics import BooleanAccuracy
from allennlp.nn import util
from train_utils.decoder_trainer import NormalizedMaximumMarginalLikelihood
from allennlp_semparse.state_machines.states.grammar_statelet import GrammarStatelet
from grammar_based_state import GrammarBasedState
from allennlp_semparse.state_machines.states.rnn_statelet import RnnStatelet
from linking_transition_function import LinkingTransitionFunction
from allennlp_semparse.fields.production_rule_field import ProductionRule
import os
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from worlds.grail_world import KBWorld
from sexpression_state import SExpressionState
from allennlp_semparse.common.errors import ParsingError
from random import choices, shuffle, choice
from train_utils.early_stop_beam_search import EarlyStopBeamSearch
import logging
from allennlp.common.checks import ConfigurationError
from sparql_executor import exec_sparql_fix_literal
from train_utils.metrics import SetF1Measure, GraphMatchMeasure, SetHit1Measure
import json
from utils import Entity, GeneralizeLevel
import random

logger = logging.getLogger(__name__)


@Model.register('grail')
class KBQAParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_encoder: Seq2SeqEncoder,
                 decoder_beam_size: int,
                 decoder_node_size: int,
                 input_attention: Attention,
                 text_embedder: TextFieldEmbedder,
                 max_decoding_steps: int,
                 action_embedding_dim: int,
                 entity_embedding_dim: int,
                 training_beam_size: int,
                 dropout_rate: float,
                 decoder_num_layers: int = 1,
                 maximum_negative_chunk: int = 5,
                 maximum_negative_cand: int = 200,
                 # maximum execution times on beam checking
                 maximum_execution_times: int = 5,
                 dynamic_negative_ratio: float = 0.5,
                 rule_namespace: str = 'rule_labels',
                 utterance_agg_method: str = 'first',
                 entity_order_method: str = 'shuffle',
                 keep_entity_always: bool = False,
                 # parser setting
                 use_type_checking: bool = False,
                 use_virtual_forward: bool = False,
                 use_entity_anchor: bool = False,
                 use_runtime_prune: bool = False,
                 use_beam_check: bool = False,
                 # parser linking and schema encoding setting
                 use_feature_score: bool = False,
                 use_schema_encoder: bool = False,
                 use_linking_embedding: bool = False,
                 # use schema as the action embedding for next state
                 use_schema_as_input: bool = False,
                 # not select with attention (i.e. not use linking scores)
                 use_attention_select: bool = True,
                 # bert setting
                 use_bert: bool = False,
                 debug_parsing: bool = False,
                 evaluate_f1: bool = False,
                 evaluate_hits1: bool = False,
                 demo_mode: bool = False,
                 # evaluate setting
                 log_eval_info_file: str = None,
                 # ends with jsonl
                 log_analysis_info_file: str = None,
                 # pass by arguments
                 fb_roles_file: str = None,
                 fb_types_file: str = None,
                 reverse_properties_file: str = None):
        super().__init__(vocab)

        self.vocab = vocab
        self.max_decoding_steps = max_decoding_steps

        self.use_type_checking = use_type_checking
        self.use_virtual_forward = use_virtual_forward
        self.use_entity_anchor = use_entity_anchor
        self.use_runtime_prune = use_runtime_prune

        # padding for invalid action
        self.action_padding_index = -1

        # dropout inside/outside LSTM
        if dropout_rate > 0:
            self.var_dropout = InputVariationalDropout(p=dropout_rate)
        else:
            self.var_dropout = lambda x: x

        self.dropout = Dropout(p=dropout_rate)

        # embedding layer of action like `Statement -> Select` and etc.
        self.rule_namespace = rule_namespace
        num_actions = vocab.get_vocab_size(self.rule_namespace)

        """
        Define encoder layer
        """
        self.text_embedder = text_embedder

        # for bert/non-bert, we use the same text encoder
        self.text_encoder = text_encoder

        self.encoder_output_dim = text_encoder.get_output_dim()
        self.embedding_dim = self.text_embedder.get_output_dim()
        self.scale = int(math.sqrt(self.embedding_dim))
        # self.scale = 1.0

        self.decoder_num_layers = decoder_num_layers

        """
        Define embedding layer
        """
        # used for scoring the action selection
        self.output_action_embedder = Embedding(num_embeddings=num_actions,
                                                embedding_dim=action_embedding_dim)
        # used for sequence generation input
        self.action_embedder = Embedding(num_embeddings=num_actions,
                                         embedding_dim=action_embedding_dim)

        # entity type embedding layer such as entity/set/relation. 0 for padding
        # TODO: entity type embedding will add in the text embedding, so it should keep the same dimension
        self.num_entity_types = 5
        self.link_entity_type_embedder = Embedding(num_embeddings=self.num_entity_types,
                                                   embedding_dim=entity_embedding_dim,
                                                   padding_index=0)

        self.output_entity_type_embedder = Embedding(num_embeddings=self.num_entity_types,
                                                     embedding_dim=action_embedding_dim,
                                                     padding_index=0)

        # Note: the dimension is highly related to the knowledge graph field.
        # please go there to see the dimensions of this linking feature.
        self.linking_layer = torch.nn.Linear(4, 1)
        torch.nn.init.uniform_(self.linking_layer.weight)
        torch.nn.init.zeros_(self.linking_layer.bias)

        # in our setting per node beam size should be larger than beam size because
        # there will be a lot of invalid actions
        self.beam_search = EarlyStopBeamSearch(beam_size=decoder_beam_size,
                                               per_node_beam_size=decoder_node_size)

        # embedding of the first special action
        self.first_action_embedding = nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self.first_attended_output = nn.Parameter(torch.FloatTensor(self.encoder_output_dim))

        # initialize parameters
        torch.nn.init.uniform_(self.first_action_embedding, -0.1, 0.1)
        torch.nn.init.uniform_(self.first_attended_output, -0.1, 0.1)

        """
        Define parsing related variants
        """
        self.use_schema_encoder = use_schema_encoder
        self.use_linking_embedding = use_linking_embedding
        self.use_schema_as_input = use_schema_as_input

        if self.use_schema_as_input:
            if self.embedding_dim != action_embedding_dim:
                self.schema_to_action = torch.nn.Linear(in_features=self.embedding_dim,
                                                        out_features=action_embedding_dim)
                torch.nn.init.uniform_(self.schema_to_action.weight)
                torch.nn.init.zeros_(self.schema_to_action.bias)
            else:
                self.schema_to_action = lambda x: x

        self.use_attention_select = use_attention_select
        self.use_feature_score = use_feature_score

        # responsible for column encoding and table encoding respectively
        if use_schema_encoder:
            self.schema_encoder = PytorchSeq2VecWrapper(nn.LSTM(input_size=self.embedding_dim,
                                                                hidden_size=int(self.embedding_dim / 2),
                                                                bidirectional=True,
                                                                batch_first=True))
        else:
            self.schema_encoder = None

        self.use_bert = use_bert
        mix_feedforward = FeedForward(input_dim=self.encoder_output_dim,
                                      num_layers=1,
                                      hidden_dims=1,
                                      activations=torch.nn.Sigmoid(),
                                      dropout=0.0)

        self.transition_function = LinkingTransitionFunction(encoder_output_dim=self.encoder_output_dim,
                                                             action_embedding_dim=action_embedding_dim,
                                                             input_attention=input_attention,
                                                             add_action_bias=False,
                                                             mixture_feedforward=mix_feedforward,
                                                             dropout=dropout_rate,
                                                             num_layers=self.decoder_num_layers,
                                                             use_attention_select=self.use_attention_select)

        """
        Define the linear layer convert matching feature into score
        """

        """
        Define metrics to measure
        """
        self._iid_split_accuracy = GraphMatchMeasure(
            fb_roles_file=fb_roles_file,
            fb_types_file=fb_types_file,
            reverse_properties_file=reverse_properties_file
        )
        self._com_split_accuracy = GraphMatchMeasure(
            fb_roles_file=fb_roles_file,
            fb_types_file=fb_types_file,
            reverse_properties_file=reverse_properties_file
        )
        self._zero_split_accuracy = GraphMatchMeasure(
            fb_roles_file=fb_roles_file,
            fb_types_file=fb_types_file,
            reverse_properties_file=reverse_properties_file
        )
        self._avg_accuracy = GraphMatchMeasure(
            fb_roles_file=fb_roles_file,
            fb_types_file=fb_types_file,
            reverse_properties_file=reverse_properties_file
        )

        self._cand_iid_accuracy = BooleanAccuracy()
        self._cand_com_accuracy = BooleanAccuracy()
        self._cand_zero_accuracy = BooleanAccuracy()

        self._avg_f1 = SetF1Measure()
        self._avg_hits1 = SetHit1Measure()
        self._iid_split_f1 = SetF1Measure()
        self._com_split_f1 = SetF1Measure()
        self._zero_split_f1 = SetF1Measure()

        """
        Debugging setting
        """
        self.debug_parsing = debug_parsing
        self.eval_file = log_eval_info_file
        if log_eval_info_file:
            if os.path.exists(self.eval_file):
                os.remove(self.eval_file)
            self.eval_file_obj = open(self.eval_file, "w", encoding="utf8")
            self.debug_template = "### qid {0}\n\n" \
                                  "**Logic Correct**: {1}\n\n" \
                                  "**Answer Correct**: {2}\n\n" \
                                  "**KB Recall**: {3}\n\n" \
                                  "**Utterance**: {4}\n\n" \
                                  "**Predict Expression**: {5}\n\n" \
                                  "**Golden  Expression**: {6}\n\n" \
                                  "**Level**    : {7}\n\n" \
                                  "**Golden Node**    : \n{8}\n\n" \
                                  "**Candidate Node** : \n{9}\n\n" \
                                  "**Golden Edge**   : \n{10}\n\n" \
                                  "**Candidate Edge** : \n{11}\n\n" \
                                  "**Golden Sp**: \n{12}\n\n" \
                                  "**Attention**: \n{13}\n\n" \
                                  "**Action Space(Prob)**: \n{14}\n\n" \
                                  "**Last Valid Action**: \n{15}\n\n"

        self.analysis_file = log_analysis_info_file
        if self.analysis_file:
            self.analysis_file_obj = open(self.analysis_file, "w", encoding="utf8")

        self.evaluate_f1 = evaluate_f1
        self.evaluate_hits1 = evaluate_hits1
        self.demo_mode = demo_mode

        """
        Define transition function
        """

        self.decoder_trainer = NormalizedMaximumMarginalLikelihood(training_beam_size)
        # maximum word piece length
        self.maximum_input_length = 512

        self.cls_word = 101

        self.dynamic_negative_ratio = dynamic_negative_ratio
        self.maximum_negative_chunk = maximum_negative_chunk
        self.maximum_negative_cand = maximum_negative_cand

        assert utterance_agg_method in ["first", "max", "mean", "shuffle"]
        # first: take the first utterance repr
        # max: take the max pooling of utterance repr
        # mean: take the average pooling of utterance repr
        # shuffle: take a random utterance repr
        self.utterance_agg_method = utterance_agg_method

        assert entity_order_method in ["alphabet", "shuffle"]
        self.entity_order_method = entity_order_method
        self.keep_entity_always = keep_entity_always
        self.use_post_beam_check = use_beam_check

        # fb_roles, fb_types and reverse_properties
        self.fb_roles_file = fb_roles_file
        self.fb_types_file = fb_types_file
        self.reverse_properties_file = reverse_properties_file

        # to reduce the execution times
        self.maximum_execution_times = maximum_execution_times

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.demo_mode:
            output_dict = {"demo": output_dict["demo"]}
        else:
            output_dict = {key: val for key, val in output_dict.items()
                           if key in ["logical_form", "answer", "qid", "utterance", "demo"]}
        return output_dict

    def make_demo_output(self, predict_answer: List[str],
                         predict_sparql: str,
                         predict_logic_form: str,
                         predict_action: List[str],
                         world: KBWorld):
        predict_answer = predict_answer[:10]
        entity_meta_info = world.entity_meta_info
        selected_schema = [action_rule.split(" -> ")[1]
                           for action_rule in predict_action]

        # ordered list for entity based on offset
        entity_list = []
        # ordered list for class and relation based on score
        class_list = []
        relation_list = []

        LINK_POINT = "/meta/{}"
        for ent in world.graph_query_info["nodes"]:
            selected = ent["id"] in selected_schema
            cls_id = ent["id"]
            if ent["node_type"] == "class":
                class_list.append({
                    "name": ent["friendly_name"],
                    "score": ent["score"],
                    "select": selected,
                    "link": LINK_POINT.format(cls_id)
                })
            else:
                ent_id = ent["id"]
                if ent_id in entity_meta_info:
                    ent_type = entity_meta_info[ent["id"]]["ent_type"]
                    entity_list.append({
                        "title": ent["friendly_name"],
                        "link": LINK_POINT.format(ent_id),
                        "id": ent["id"],
                        "type": ent_type,
                        "class": ent["class"],
                        "start": ent["offset"][0],
                        "end": ent["offset"][1],
                        "score": ent["score"],
                        "select": selected,
                    })
                else:
                    # literal entity
                    entity_list.append({
                        "title": ent["friendly_name"],
                        "id": ent["id"],
                        "type": "literal",
                        "class": ent["class"],
                        "start": ent["offset"][0],
                        "end": ent["offset"][1],
                        "score": 0.0,
                        "select": selected
                    })

        for rel in world.graph_query_info["edges"]:
            selected = rel["relation"] in selected_schema
            rel_id = rel["relation"]
            relation_list.append({
                "name": rel["friendly_name"],
                "score": rel["score"],
                "select": selected,
                "link": LINK_POINT.format(rel_id)
            })

        entity_list = sorted(entity_list, key=lambda x: (x["start"], - x["score"]))
        class_list = sorted(class_list, key=lambda x: x["score"], reverse=True)
        relation_list = sorted(relation_list, key=lambda x: x["score"], reverse=True)

        # class_list = class_list[:50]
        # relation_list = relation_list[:50]
        # split tokens, and construct entity candidate
        entity_tokens = world.entity_offset_tokens
        entity_pointer = 0
        entity_bound = len(entity_list)
        entity_exist = [False] * len(entity_tokens)
        last_token_ind = 0
        candidate_list = []

        # overlap index
        while entity_pointer < entity_bound:
            cur_ent = entity_list[entity_pointer]
            next_token_ind = cur_ent["start"]
            # if exist, continue
            if entity_exist[next_token_ind]:
                entity_pointer += 1
                continue
            if next_token_ind != last_token_ind:
                candidate_list.append({
                    "text": " ".join(entity_tokens[last_token_ind: next_token_ind]),
                    "type": "text"
                })
            last_token_ind = cur_ent["end"]
            # set status
            for i in range(next_token_ind, last_token_ind):
                entity_exist[i] = True
            # literal entity
            if "link" not in cur_ent:
                candidate_list.append({
                    "text": " ".join(entity_tokens[next_token_ind: last_token_ind]),
                    "type": "literal",
                    "id": cur_ent["id"],
                    "ent_type": "literal",
                    "title": cur_ent["title"],
                    "index": entity_pointer,
                    "select": cur_ent["select"]
                })
            else:
                candidate_list.append({
                    "text": " ".join(entity_tokens[next_token_ind: last_token_ind]),
                    "type": "entity",
                    "id": cur_ent["id"],
                    "ent_type": cur_ent["type"],
                    "title": cur_ent["title"],
                    "link": cur_ent["link"],
                    "index": entity_pointer,
                    "select": cur_ent["select"]
                })
            entity_pointer += 1

        candidate_list.append({
            "text": " ".join(entity_tokens[last_token_ind:]),
            "type": "text"
        })

        # take top 20, and normalize the score into 100 percents
        top_class_score = class_list[0]["score"]
        bottom_class_score = class_list[-1]["score"]
        top_relation_score = relation_list[0]["score"]
        bottom_relation_score = relation_list[-1]["score"]

        for example in class_list:
            example["score"] = int(100 * (example["score"] - bottom_class_score) / (top_class_score - bottom_class_score + 0.1))
        for example in relation_list:
            example["score"] = int(100 * (example["score"] - bottom_relation_score) / (top_relation_score - bottom_relation_score + 0.1))

        data_sample = {
            "answer": predict_answer,
            "entity_candidate": candidate_list,
            "schema_class": class_list,
            "schema_relation": relation_list,
            "logic_form": predict_logic_form,
            "sparql": predict_sparql
        }

        return data_sample

    def build_analysis_info(self,
                            predict_logic_forms: List[str],
                            predict_answers: List[List[str]],
                            predict_actions: List[List[str]],
                            debug_info: List[Dict],
                            scores: List[float],
                            worlds: List[KBWorld],
                            logic_form_correct: List[bool],
                            answer_correct: List[bool]) -> List[Dict]:
        analysis_dict = []
        for i, world in enumerate(worlds):
            # assume the result only batch 1
            qid = world.world_id
            # only take top 10 for space consideration
            ground_answer = world.answers[:10]
            predict_answer = predict_answers[i][:10]
            ground_sparql = world.sparql_query
            sentence = world.origin_utterance
            predict_logic_form = predict_logic_forms[i]
            best_predict_action = predict_actions[i]
            action_flatten = ", ".join(best_predict_action)

            predict_entity = []
            for action in best_predict_action:
                lhs, rhs = action.split(' -> ')
                if lhs == Entity.__name__:
                    predict_entity.append(rhs)

            entity_dict = set()
            class_dict = set()
            relation_dict = set()
            for ent in world.graph_query_info["nodes"]:
                if ent["node_type"] == "class":
                    class_dict.add(ent["id"])
                else:
                    entity_dict.add(ent["id"])
            for rel in world.graph_query_info["edges"]:
                relation_dict.add(rel["relation"])

            for cand_ent in world.kb_context.entity_list:
                if cand_ent.entity_id in class_dict:
                    class_dict.remove(cand_ent.entity_id)
                elif cand_ent.entity_id in entity_dict:
                    entity_dict.remove(cand_ent.entity_id)
            for can_rel in world.kb_context.relation_list:
                if can_rel.relation_class in relation_dict:
                    relation_dict.remove(can_rel.relation_class)

            analysis_dict.append({
                "qid": qid,
                "question": sentence,
                "action": action_flatten,
                "entity": predict_entity,
                "predict_sexpression": predict_logic_form,
                "ground_sexpression": world.sexpression_eval,
                "predict_answer": predict_answer,
                "ground_answer": ground_answer,
                "ground_sparql": ground_sparql,
                "entity_recall": True if len(entity_dict) == 0 else "; ".join(entity_dict),
                "class_recall": True if len(class_dict) == 0 else "; ".join(class_dict),
                "relation_recall": True if len(relation_dict) == 0 else "; ".join(relation_dict),
                "debug_info": debug_info[i],
                "logic_form_correct": logic_form_correct[i],
                "answer_correct": answer_correct[i],
                "score": scores[i],
                "level": world.level
            })
        return analysis_dict

    @overrides
    def forward(self,
                utterance: Dict[str, Dict[str, torch.LongTensor]],
                valid_actions: List[List[ProductionRule]],
                worlds: List[KBWorld],
                schema: Dict[str, torch.LongTensor],
                entity_type: torch.LongTensor,
                # 1,2,3 for i.i.d., compositional and zero-shot
                dataset_type: torch.LongTensor,
                # Action sequence with copy is built for copy segment.
                # Towards the first turn, it is equal to action_sequence
                action_sequence: torch.LongTensor = None,
                semantic_cons: List[Dict] = None,
                literal_relation: List[Set] = None,
                anchor_cons: List[Dict] = None,
                ground_entity_start_end: torch.LongTensor = None,
                ground_relation_start_end: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        device = entity_type.device

        if 'tokens' in utterance:
            assert self.use_bert == False
            batch_size = utterance['tokens']['token_ids'].size()[0]
        else:
            batch_size = utterance['bert']['token_ids'].size()[0]

        try:
            # batch_size x col_size (we should expand it into inter_size then)
            with torch.set_grad_enabled(self.training):
                entity_type = entity_type.long().view(batch_size, -1)
                entity_domain = [world.kb_context.knowledge_graph.entity_domain for world in worlds]
                encoder_input, encoder_mask, linked_scores, encoding_schema = self._init_parser_input(
                    utterance=utterance,
                    schema=schema,
                    entity_type=entity_type,
                    entity_domain=entity_domain,
                    ground_entity_start_end=ground_entity_start_end,
                    ground_relation_start_end=ground_relation_start_end)

                initial_state = self._init_grammar_state(encoder_input,
                                                         encoder_mask,
                                                         batch_size,
                                                         worlds,
                                                         semantic_cons,
                                                         anchor_cons,
                                                         literal_relation,
                                                         linked_scores,
                                                         valid_actions,
                                                         entity_type,
                                                         encoding_schema)

            if action_sequence is not None:
                # Remove the trailing dimension (from ListField[ListField[IndexField]]).
                action_sequence = action_sequence.squeeze(-1)

            if self.training:
                action_mask = torch.ne(action_sequence, self.action_padding_index)
                decode_output = self.decoder_trainer.decode(initial_state,
                                                            self.transition_function,
                                                            # expand the dimension of `beam`
                                                            (action_sequence.unsqueeze(dim=1),
                                                             action_mask.unsqueeze(dim=1)))
                return {'loss': decode_output['loss']}
            else:
                loss = torch.tensor([0]).float().to(device)
                outputs: Dict[str, Any] = {
                    'loss': loss
                }

                # In evaluation, segment-level copy will lead to two concerns:
                # 1. the evaluation of turn $t$ can only be done after the turn $t-1$, so we need dynamically update
                #    precedent action sequence stored in the world.
                # 2. the generating results should be reformulated as non-copy existing (e.g. expand the copy action).
                num_steps = self.max_decoding_steps
                # construct kb_contexts

                # construct action mapping
                action_mappings: List[List[str]] = [[production_rule[0] for production_rule in valid_action]
                                                    for valid_action in valid_actions]

                # This tells the state to start keeping track of debug info, which we'll pass along in
                # our output dictionary.
                initial_state.debug_info = [[] for _ in range(batch_size)]

                best_final_states = self.beam_search.search(num_steps,
                                                            initial_state,
                                                            self.transition_function,
                                                            keep_final_unfinished_states=False)

                best_beam_ids = [0] * batch_size

                if self.use_post_beam_check:
                    # use sparql to validate if the logical form is valid
                    for i in range(batch_size):
                        world = worlds[i]
                        if i in best_final_states:
                            maximum_execution_times = min(len(best_final_states[i]), self.maximum_execution_times)
                            for best_idx in range(maximum_execution_times):
                                try:
                                    action_seq = [str(action_mappings[i][action_id]) for action_id in
                                                  best_final_states[i][best_idx].action_history[0]]
                                    logic_form = self.predict_sexpression(action_seq=action_seq,
                                                                          world=world)
                                    sparql_query = world.sparql_converter(logic_form)
                                    answer, status_code = exec_sparql_fix_literal(sparql_query)
                                    if status_code == 200 and len(answer) != 0:
                                        best_beam_ids[i] = best_idx
                                        break
                                except Exception or ParsingError or TypeError:
                                    continue

                outputs['debug_info'] = [best_final_states[i][best_beam_ids[i]].debug_info[0]
                                         if i in best_final_states else [] for i in range(batch_size)]

                action_predict = [[str(action_mappings[i][action_id])
                                   for action_id in
                                   best_final_states[i][best_beam_ids[i]].action_history[0]]
                                  if i in best_final_states else []
                                  for i in range(batch_size)]
                # in test mode, predict the actual SQL
                outputs['best_predict_action'] = action_predict

                outputs['best_predict_score'] = [float(best_final_states[i][best_beam_ids[i]].score[0])
                                                 if i in best_final_states else 0.0
                                                 for i in range(batch_size)]
                # add utterance for better reading
                outputs['utterance'] = [world.origin_utterance for world in worlds]

                for i in range(batch_size):
                    # every sample is a list
                    debug_sample: Dict = outputs['debug_info'][i]
                    for info_dict in debug_sample:
                        info_dict['question_attention'] = ["{0:.2f}".format(float(num))
                                                           for num in info_dict['question_attention']]
                        info_dict['considered_actions'] = [action_mappings[i][action_id]
                                                           for action_id in info_dict['considered_actions']]
                        info_dict['probabilities'] = [float(prob) for prob in info_dict['probabilities']]

                sexpression_predict = []
                for action_seq, world in zip(outputs['best_predict_action'], worlds):
                    if len(action_seq) != 0:
                        try:
                            generated_sexpression = self.predict_sexpression(action_seq=action_seq,
                                                                             world=world)
                        except ParsingError:
                            # if fail, return one valid SQL
                            sexpression_predict.append("")
                            exec_info = sys.exc_info()
                            traceback.print_exception(*exec_info)
                        else:
                            sexpression_predict.append(generated_sexpression)
                    else:
                        sexpression_predict.append("")

                outputs['best_predict_sexpression'] = sexpression_predict
                sexpression_gold: List[str] = [world.sexpression_eval for world in worlds]
                answer_gold = [world.answers for world in worlds]

                sparql_predict = []
                if self.evaluate_f1 or self.evaluate_hits1:
                    answer_predict = []
                    for i, logic_form in enumerate(sexpression_predict):
                        try:
                            if self._avg_accuracy.equals(logic_form, sexpression_gold[i]):
                                answer = answer_gold[i]
                            elif logic_form != '':
                                sparql_query = worlds[i].sparql_converter(logic_form)
                                sparql_predict.append(sparql_query)
                                answer = exec_sparql_fix_literal(sparql_query)[0]
                            else:
                                answer = []
                        except Exception as e:
                            print(e)
                            print("Error on execute logic form: {}".format(logic_form))
                            answer = []
                        answer_predict.append(answer)
                else:
                    answer_predict = [[] for _ in range(batch_size)]

                outputs["logical_form"] = sexpression_predict
                outputs["answer"] = answer_predict
                outputs["qid"] = [world.world_id for world in worlds]

                if action_sequence is not None:
                    # evaluate sexpression accuracy
                    gold_labels = torch.ones(batch_size, device=device)
                    recall_labels = []
                    for i in range(batch_size):
                        # at least one element not equal to -1
                        if torch.nonzero(action_sequence[i].ne(self.action_padding_index)).flatten().size()[0] != 0:
                            recall_labels.append(1)
                        else:
                            recall_labels.append(0)
                    recall_labels = torch.tensor(recall_labels, device=gold_labels.device, dtype=torch.long)

                    # dataset type number is correlated to dataset reader
                    self._avg_accuracy(sexpression_predict, sexpression_gold)
                    self._iid_split_accuracy(sexpression_predict, sexpression_gold,
                                             dataset_type == GeneralizeLevel.iid)
                    self._com_split_accuracy(sexpression_predict, sexpression_gold,
                                             dataset_type == GeneralizeLevel.com)
                    self._zero_split_accuracy(sexpression_predict, sexpression_gold,
                                              dataset_type == GeneralizeLevel.zero)

                    self._cand_iid_accuracy(recall_labels, gold_labels, dataset_type == GeneralizeLevel.iid)
                    self._cand_com_accuracy(recall_labels, gold_labels, dataset_type == GeneralizeLevel.com)
                    self._cand_zero_accuracy(recall_labels, gold_labels, dataset_type == GeneralizeLevel.zero)

                    self._avg_f1(answer_predict, answer_gold)
                    self._avg_hits1(answer_predict, answer_gold)
                    self._iid_split_f1(answer_predict, answer_gold, dataset_type == GeneralizeLevel.iid)
                    self._com_split_f1(answer_predict, answer_gold, dataset_type == GeneralizeLevel.com)
                    self._zero_split_f1(answer_predict, answer_gold, dataset_type == GeneralizeLevel.zero)

                    logic_form_correctness = [self._avg_accuracy.equals(sexpression_predict[i],
                                                                        sexpression_gold[i])
                                              for i in range(batch_size)]
                    answer_correctness = [self._avg_f1.equals(answer_predict[i],
                                                              answer_gold[i])
                                          for i in range(batch_size)]
                    if self.analysis_file:
                        output_dict_list = self.build_analysis_info(sexpression_predict,
                                                                    answer_predict,
                                                                    action_predict,
                                                                    outputs['debug_info'],
                                                                    outputs['best_predict_score'],
                                                                    worlds,
                                                                    logic_form_correctness,
                                                                    answer_correctness)
                        for output_dict in output_dict_list:
                            self.analysis_file_obj.write(json.dumps(output_dict) + "\n")

                    if self.eval_file:
                        for i in range(batch_size):
                            # only record which recall could cover but our algorithm fails to predict
                            logic_form_correct = logic_form_correctness[i]
                            answer_correct = answer_correctness[i]
                            recall_correct = recall_labels[i] == 1

                            hit_ids = []
                            for action_str in action_predict[i]:
                                lhs, rhs = action_str.split(' -> ')
                                hit_ids.append(rhs)

                            predict_node_path = ""
                            for entity in worlds[i].kb_context.entity_list:
                                if entity.entity_id in hit_ids:
                                    predict_node_path += "Node: {} -> {} -> {} -> {}\n".format(entity.entity_type,
                                                                                               entity.entity_class,
                                                                                               entity.entity_id,
                                                                                               entity.text)
                            predict_edge_path = ""
                            for edge in worlds[i].kb_context.relation_list:
                                if edge.relation_id in hit_ids:
                                    predict_edge_path += "Rel: {} -> {}\n".format(edge.relation_id,
                                                                                  edge.text)
                            node_path = ""
                            edge_path = ""
                            if worlds[i].graph_query_info:
                                for node in worlds[i].graph_query_info["nodes"]:
                                    node_path += "Node({}): {} -> {} -> {} -> {}\n".format(node['nid'],
                                                                                           node['node_type'],
                                                                                           node['class'],
                                                                                           node['id'],
                                                                                           node['friendly_name'])
                                for edge in worlds[i].graph_query_info["edges"]:
                                    edge_path += "Rel({}-{}): {} -> {}\n".format(edge['start'], edge['end'],
                                                                                 edge['relation'],
                                                                                 edge['friendly_name'])

                            attention_debug_info = ""
                            utter_token_ids = utterance['bert']['token_ids'][i].cpu().tolist()
                            attention_utter = [self.vocab._index_to_token['tags'][ind] for ind in utter_token_ids]
                            # only print the trace of best prediction
                            for step_ind in range(len(outputs['debug_info'][i])):
                                example_debug_info = outputs['debug_info'][i][step_ind]
                                attention_debug_info += "\nStep: {} ".format(step_ind + 1)
                                attention_debug_info += "Action: {} ".format(action_predict[i][step_ind])
                                attention_debug_info += "Attention: "
                                for j in range(len(attention_utter)):
                                    attn_score = example_debug_info['question_attention'][j]
                                    attention_debug_info += attention_utter[j] + "/" + attn_score + " "

                            action_debug_info = ""
                            for step_ind in range(len(outputs['debug_info'][i])):
                                example_debug_info = outputs['debug_info'][i][step_ind]
                                action_debug_info += "\nStep: {} ".format(step_ind + 1)
                                hit_indices = [tup[0] for tup in sorted([(ind, score) for ind, score
                                                                         in
                                                                         enumerate(example_debug_info['probabilities'])
                                                                         if score >= 0.01], key=lambda x: x[1],
                                                                        reverse=True)]
                                record_actions = ["{0}({1:.2f})".format(example_debug_info['considered_actions'][ind],
                                                                        example_debug_info['probabilities'][ind])
                                                  for ind in hit_indices]
                                action_debug_info += ", ".join(record_actions)

                            last_valid_action_info = ""
                            if self.beam_search.last_valid_state and not (logic_form_correct or answer_correct):
                                last_state: GrammarBasedState = self.beam_search.last_valid_state
                                state_len = len(last_state.batch_indices)
                                state_counter = 0
                                for state_ind in range(state_len):
                                    # must belong to this batch, we can print it
                                    batch_ind = last_state.batch_indices[state_ind]
                                    if batch_ind != i:
                                        continue
                                    state_counter += 1
                                    cur_action_seq = ", ".join([str(action_mappings[i][action_id])
                                                                for action_id in last_state.action_history[state_ind]])
                                    cur_score = "{:.2f}".format(float(last_state.score[state_ind]))
                                    last_valid_action_info += "\nCandidate {}".format(state_counter)
                                    last_valid_action_info += "\n   Action: {}".format(cur_action_seq)
                                    last_valid_action_info += "\n   Score:  {}\n".format(cur_score)

                            self.eval_file_obj.write(
                                self.debug_template.format(worlds[i].world_id,
                                                           logic_form_correct,
                                                           answer_correct,
                                                           recall_correct,
                                                           worlds[i].origin_utterance,
                                                           sexpression_predict[i] + " (" +
                                                           "{:.2f}".format(outputs['best_predict_score'][i]) + ")",
                                                           sexpression_gold[i],
                                                           worlds[i].level,
                                                           node_path,
                                                           predict_node_path,
                                                           edge_path,
                                                           predict_edge_path,
                                                           worlds[i].sparql_query,
                                                           attention_debug_info,
                                                           action_debug_info,
                                                           last_valid_action_info) + "\n")
                if self.demo_mode:
                    if len(action_predict) and len(sparql_predict):
                        outputs['demo'] = [self.make_demo_output(
                            predict_logic_form=sexpression_predict[0],
                            predict_sparql=sparql_predict[0],
                            predict_answer=answer_predict[0],
                            predict_action=action_predict[0],
                            world=worlds[0]
                        )]
                    else:
                        outputs['demo'] = [self.make_demo_output(
                            predict_logic_form="",
                            predict_sparql="",
                            predict_answer=[],
                            predict_action=[],
                            world=worlds[0]
                        )]
            return outputs
        except RuntimeError as err:
            logging.error(err)
            return {
                'loss': torch.tensor(0.0, requires_grad=True, device=device)
            }

    def predict_sexpression(self, action_seq, world: KBWorld) -> str:
        if len(action_seq) == self.max_decoding_steps:
            # TODO: somehow hacky, but we cannot find a elegant way to catch the recursive ParsingError
            raise ParsingError("Reach the maximum decoding step, parsing error")
        generated_sexpression = world.language.action_sequence_to_logical_form(action_seq)
        post_process_sexpression = world.postprocess(generated_sexpression)
        return post_process_sexpression

    def _chunking_joint_utterance_and_entity(self, utterance: Dict[str, torch.Tensor],
                                             select_entity_index: List[List[int]],
                                             schema_text: Dict[str, torch.Tensor],
                                             max_ent_size: int,
                                             batch_size: int,
                                             # BERT will prepend [CLS] and [SEP] on entity
                                             # here we drop the [CLS] for longer capacity
                                             drop_first_token: bool = True):
        """
        In training, we want to select some entity as negative candidates and only encode them within
        utterance via BERT encoding to speed up training without performance drop.
        Here we follow the process:
        1. sample selected entity indexes, then shuffle and record their positions in text (maybe cross batch);
        2. for unselected entity indexes, set their default schema position as the first [CLS] flag in utterance;
        3. organize these schema positions into [max_ent_size x 3] tensor to be compatible with the remaining code.

        In development, we must select all entity candidates and encode them by chunking, but interface is the same
        :return:
        schema_position: [max_ent_size x 3], which records the chunk/start/end position of each entity in
        order of entities in the provided knowledge graph (IMPORTANT)
        chunk_utterance: Dict[str, torch.LongTensor], which combines utterance and schema text to build new states
        """
        # utterance/schema_text:
        # token_ids: batch_size x max_ent_size x ent_token_size
        # mask: batch_size x max_ent_size x ent_token_size
        # type_ids: batch_size x max_ent_size x ent_token_size
        # segment_concat_mask: batch_size x max_ent_size x ent_token_size
        # NOTE: in our scenario, segment_concat_mask is exactly same with mask

        schema_position = torch.zeros((batch_size, max_ent_size, 3), device=utterance['token_ids'].device).long()
        schema_local_position = torch.zeros((batch_size, max_ent_size), device=utterance['token_ids'].device).long()
        # one utterance may correspond to multiple chunks
        chunk_token_ids = []
        chunk_count_start = 0
        utterance_len_list = []
        batch_to_chunk = {}
        for i in range(batch_size):
            # set default schema position
            remain_space = self.maximum_input_length - int(utterance['mask'][i].sum())
            utterance_no_pad = utterance['token_ids'][i][utterance['mask'][i]]
            utterance_len = len(utterance_no_pad)
            """
            combine utterance and entity tokens 
            """
            cur_sel_entity = select_entity_index[i]
            # take the corresponding entity tokens
            cand_entity_token_ids = schema_text['token_ids'][i, cur_sel_entity, :]
            # sel_ent_size x ent_token_size
            cand_entity_token_mask = schema_text['mask'][i, cur_sel_entity, :]
            # sel_ent_size
            cand_entity_token_len = cand_entity_token_mask.sum(dim=1)
            cand_no_pad_entity_ids = cand_entity_token_ids[cand_entity_token_mask]
            # remove the last [SEP]
            if drop_first_token:
                cand_no_pad_entity_ids = cand_no_pad_entity_ids[cand_no_pad_entity_ids != self.cls_word]
                # drop the last one
                cand_entity_token_len = cand_entity_token_len - 1
            # compute chunk size
            chunk_size = math.ceil(len(cand_no_pad_entity_ids) / remain_space)

            assert cand_no_pad_entity_ids.size(0) == cand_entity_token_len.sum()

            """
            compute schema position 
            """
            cand_entity_cumsum_len = cand_entity_token_len.cumsum(dim=0)
            utterance_len_list.extend([utterance_len] * chunk_size)
            # record batch_to_chunk
            batch_to_chunk[i] = [chunk_count_start + i for i in range(chunk_size)]
            # initialize chunk token ids
            if chunk_size == 1:
                combined_token_ids = torch.cat([utterance_no_pad, cand_no_pad_entity_ids])
                chunk_token_ids.append(combined_token_ids)

                for entity_ind_in_select in range(0, len(cand_entity_token_len)):
                    entity_ind_in_world = cur_sel_entity[entity_ind_in_select]
                    entity_ind_in_chunk = entity_ind_in_select
                    schema_position[i, entity_ind_in_world][0] = chunk_count_start
                    # only one chunk, so we use 0 always
                    schema_local_position[i, entity_ind_in_world] = 0
                    if entity_ind_in_chunk == 0:
                        schema_position[i, entity_ind_in_world][1] = utterance_len
                    else:
                        schema_position[i, entity_ind_in_world][1] = cand_entity_cumsum_len[entity_ind_in_chunk - 1] \
                                                                     + utterance_len
                    schema_position[i, entity_ind_in_world][2] = cand_entity_cumsum_len[entity_ind_in_chunk] \
                                                                 + utterance_len - 1
            else:
                ent_borders = [0]
                token_borders = [0]
                # if cand_entity_token_len is [1,2,3,4], the cumsum is as [1,3,6,10]
                maximum_entity_len = self.maximum_input_length - utterance_len
                while True:
                    next_chunk_first_ent = int(cand_entity_cumsum_len.lt(maximum_entity_len).nonzero().max()) + 1
                    # the min index is truncated, subtraction 1 means index shifting
                    token_borders.append(int(cand_entity_cumsum_len[next_chunk_first_ent - 1])
                                         + token_borders[-1])
                    ent_borders.append(next_chunk_first_ent + ent_borders[-1])
                    if cand_entity_cumsum_len[-1] < maximum_entity_len:
                        break
                    # calculate next border
                    cand_entity_cumsum_len = cand_entity_token_len[ent_borders[-1]:].cumsum(dim=0)

                for chunk_ind in range(chunk_size):
                    # assign schema position in entity
                    chunk_token_start = token_borders[chunk_ind]
                    chunk_token_end = token_borders[chunk_ind + 1]
                    combined_token_ids = torch.cat([utterance_no_pad,
                                                    cand_no_pad_entity_ids[chunk_token_start: chunk_token_end]])
                    chunk_token_ids.append(combined_token_ids)

                    cand_entity_cumsum_len = cand_entity_token_len[ent_borders[chunk_ind]:
                                                                   ent_borders[chunk_ind + 1]].cumsum(dim=0)
                    chunk_ind_global = chunk_count_start + chunk_ind
                    for entity_ind_in_select in range(ent_borders[chunk_ind], ent_borders[chunk_ind + 1]):
                        entity_ind_in_world = cur_sel_entity[entity_ind_in_select]
                        entity_ind_in_chunk = entity_ind_in_select - ent_borders[chunk_ind]
                        schema_position[i, entity_ind_in_world][0] = chunk_ind_global
                        # local chunk ind
                        schema_local_position[i, entity_ind_in_world] = chunk_ind
                        if entity_ind_in_chunk == 0:
                            schema_position[i, entity_ind_in_world][1] = utterance_len
                        else:
                            schema_position[i, entity_ind_in_world][1] = cand_entity_cumsum_len[
                                                                             entity_ind_in_chunk - 1] + utterance_len
                        # TODO: here we subtract 1 to avoid encode [SEP] into LSTM
                        schema_position[i, entity_ind_in_world][2] = cand_entity_cumsum_len[
                                                                         entity_ind_in_chunk] + utterance_len - 1

            chunk_count_start += chunk_size
        # construct padding chunk tokens
        padding_chunk_token_ids = pad_sequence(chunk_token_ids, batch_first=True)
        chunk_max_len = padding_chunk_token_ids.size(1)
        utterance_len_tensor = torch.tensor(utterance_len_list, device=padding_chunk_token_ids.device).long()
        padding_chunk_type_ids = util.get_mask_from_sequence_lengths(utterance_len_tensor,
                                                                     max_length=chunk_max_len).long()
        chunk_token_lens = torch.tensor([len(actual_seq) for actual_seq in chunk_token_ids],
                                        device=padding_chunk_token_ids.device).long()
        padding_chunk_mask = util.get_mask_from_sequence_lengths(chunk_token_lens, max_length=chunk_max_len)
        dynamic_utterance = {
            'bert': {
                'token_ids': padding_chunk_token_ids,
                'mask': padding_chunk_mask,
                'type_ids': padding_chunk_type_ids,
                'segment_concat_mask': padding_chunk_mask
            }
        }
        return batch_to_chunk, utterance_len_list, dynamic_utterance, schema_position, schema_local_position

    def _init_parser_input(self, utterance: Dict[str, Dict[str, torch.LongTensor]],
                           schema: Dict[str, torch.LongTensor],
                           entity_type: torch.LongTensor,
                           entity_domain: List[List[str]],
                           ground_entity_start_end: torch.LongTensor = None,
                           ground_relation_start_end: torch.LongTensor = None):
        device = entity_type.device
        # {'text': {'token': tensor}, 'linking': tensor }
        # batch_size x inter_size x expect_ent_size x col_token_size

        if self.use_bert:
            all_entity_ids = schema['text']['bert']['token_ids']
            batch_size, max_ent_size, ent_word_piece_size = all_entity_ids.size()

            # batch_size x expect_ent_size x 2
            if ground_entity_start_end is not None:
                ground_entity_start_end = ground_entity_start_end.long()
                ground_relation_start_end = ground_relation_start_end.long()

            sample_ent_size = min(self.maximum_negative_cand,
                                  int(self.dynamic_negative_ratio * max_ent_size))

            select_entity_list = []
            real_ent_size_list = []

            # drop [CLS] for each schema token
            drop_first_token = True
            for i in range(batch_size):
                entity_start = int(ground_entity_start_end[i, 0])
                entity_end = int(ground_entity_start_end[i, 1])
                relation_start = int(ground_relation_start_end[i, 0])
                relation_end = int(ground_relation_start_end[i, 1])
                # record real entity size
                real_ent_size = int(schema['text']['bert']['mask'][i].sum(dim=1).nonzero(as_tuple=False).max()) + 1
                real_ent_size_list.append(real_ent_size)
                if self.training:
                    ground_entity_range = list(range(entity_start, entity_end))
                    ground_relation_range = list(range(relation_start, relation_end))
                    # maximum negative entity
                    dynamic_negative_size = min(sample_ent_size +
                                                len(ground_entity_range) +
                                                len(ground_relation_range),
                                                real_ent_size)
                    candidate_range = list(range(real_ent_size))
                    # find negative sampling index
                    negative_entity = choices(candidate_range, k=dynamic_negative_size)
                    negative_entity = [ind for ind in negative_entity if ind not in ground_entity_range]
                    negative_entity = [ind for ind in negative_entity if ind not in ground_relation_range]
                    negative_entity = negative_entity[:sample_ent_size]

                    # use chunk to truncate it
                    neg_cand_entity_token_mask = schema['text']['bert']['mask'][i, negative_entity, :]
                    pos_cand_entity_token_mask = schema['text']['bert']['mask'][i, ground_entity_range +
                                                                                   ground_relation_range, :]
                    # sel_ent_size
                    neg_cand_entity_token_len = neg_cand_entity_token_mask.sum(dim=1)
                    pos_cand_entity_token_len = pos_cand_entity_token_mask.sum(dim=1)
                    if drop_first_token:
                        neg_cand_entity_token_len -= 1
                        pos_cand_entity_token_len -= 1
                    # summarize pos_cand_entity total
                    pos_total_len = pos_cand_entity_token_len.sum()
                    utt_total_len = self.maximum_negative_chunk * utterance['bert']['mask'][i].sum()
                    max_pad_len = self.maximum_negative_chunk * (schema['text']['bert']['mask'][i].sum(dim=1).max())
                    neg_max_space = self.maximum_negative_chunk * self.maximum_input_length - pos_total_len \
                                    - utt_total_len - max_pad_len
                    neg_cand_entity_cumsum_len = neg_cand_entity_token_len.cumsum(dim=0)
                    if len(neg_cand_entity_cumsum_len) and neg_cand_entity_cumsum_len[-1] > neg_max_space:
                        # select the one which can pass the requirement
                        max_ind = int((neg_cand_entity_cumsum_len < neg_max_space).nonzero().max())
                        negative_entity = negative_entity[:max_ind + 1]

                    # in training, we only need dynamic negative candidates
                    select_entity = ground_entity_range + ground_relation_range + negative_entity
                else:
                    # but in development, we need more
                    select_entity = list(range(real_ent_size))

                # shuffle only occurred in training
                if self.entity_order_method == 'shuffle':
                    # fix seed to reproduce
                    if not self.training:
                        random.seed(1)
                    shuffle(select_entity)
                elif self.entity_order_method == 'alphabet':
                    select_entity_domain = [entity_domain[i][idx] for idx in select_entity]
                    sort_select_index = sorted(range(len(select_entity_domain)),
                                               key=select_entity_domain.__getitem__)
                    select_entity = list(map(lambda x: select_entity[x], sort_select_index))

                # add entity index into list
                select_entity_list.append(select_entity)

            # utter_end_indices: chunk_size
            # schema_position: chunk_size x expect_ent_size x 3 (chunk_ind, start, end)
            batch_to_chunk, utter_end_indices, dynamic_utterance, schema_position, schema_local_position = \
                self._chunking_joint_utterance_and_entity(utterance['bert'],
                                                          select_entity_list,
                                                          schema['text']['bert'],
                                                          max_ent_size,
                                                          batch_size,
                                                          drop_first_token)
            max_ent_token_size = (schema_position[:, :, 2] - schema_position[:, :, 1]).max()
            embedded_mix = self.text_embedder(dynamic_utterance)
            mask_mix = dynamic_utterance['bert']['mask']
            embedded_mix = embedded_mix * mask_mix.unsqueeze(dim=2).float()

            embedded_mix_dim = embedded_mix.size(-1)

            # split embedded mix into two parts: utterance & schema
            embedded_utterance = []
            encoder_mask = []
            embedded_schema = []
            # here we want to calculate the similarity between BERT utt and BERT schema
            # but we should guarantee they are in the same chunk
            # else there will be self-attention like bias !
            chunk_embedded_utterance = []
            encoder_schema_mask = []

            utt_len = max(utter_end_indices)
            for ind in range(batch_size):
                batch_map_chunk_ids = batch_to_chunk[ind]
                any_chunk_ind = batch_map_chunk_ids[0]
                # take any chunk is okay
                end_ind = utter_end_indices[any_chunk_ind]
                current_embedded_utt = []
                for chunk_ind in batch_map_chunk_ids:
                    current_embedded_utt.append(embedded_mix[chunk_ind, :end_ind, :])
                current_embedded_utt = torch.stack(current_embedded_utt)

                # chunk_size x utt_len x embedding_dim
                pad_len = utt_len - end_ind
                chunk_embedded_utterance.append(F.pad(current_embedded_utt,
                                                      pad=[0, 0, 0, pad_len],
                                                      mode='constant'))
                # fetch maximum tensor
                if self.utterance_agg_method == "first":
                    current_embedded_utt = current_embedded_utt[0]
                elif self.utterance_agg_method == "max":
                    current_embedded_utt = torch.max(current_embedded_utt, dim=0)[0]
                elif self.utterance_agg_method == "mean":
                    current_embedded_utt = torch.mean(current_embedded_utt, dim=0)
                elif self.utterance_agg_method == "shuffle":
                    if self.training:
                        random_index = choice(range(len(batch_map_chunk_ids)))
                    else:
                        # evaluate with no randomness
                        random_index = 0
                    current_embedded_utt = current_embedded_utt[random_index]
                else:
                    raise ConfigurationError(
                        "We do not support current aggregation method as: {}".format(self.utterance_agg_method))

                embedded_utterance.append(current_embedded_utt)
                encoder_mask.append(mask_mix[any_chunk_ind, :end_ind])

                cur_embedded_schema = []
                cur_schema_mask = []

                real_ent_size = real_ent_size_list[ind]
                for ent_ind in range(real_ent_size):
                    chunk_ind = schema_position[ind, ent_ind, 0]
                    entity_start_ind = schema_position[ind, ent_ind, 1]
                    entity_end_ind = schema_position[ind, ent_ind, 2]
                    pad_len = int(max_ent_token_size - (entity_end_ind - entity_start_ind))
                    # entities which are not negative ones
                    if entity_end_ind - entity_start_ind == 0:
                        # ent_size x 0.0
                        cur_embedded_schema.append(torch.zeros((pad_len, embedded_mix_dim),
                                                               device=device))
                        cur_schema_mask.append(torch.zeros(pad_len,
                                                           device=device).bool())
                    else:
                        # padding for concat
                        cur_embedded_schema.append(F.pad(embedded_mix[chunk_ind, entity_start_ind: entity_end_ind, :],
                                                         pad=[0, 0, 0, pad_len],
                                                         mode='constant'))
                        cur_schema_mask.append(F.pad(mask_mix[chunk_ind, entity_start_ind: entity_end_ind],
                                                     pad=[0, pad_len]))
                cur_embedded_schema = torch.stack(cur_embedded_schema, dim=0)
                embedded_schema.append(cur_embedded_schema)
                cur_schema_mask = torch.stack(cur_schema_mask, dim=0)
                encoder_schema_mask.append(cur_schema_mask)

            embedded_utterance = pad_sequence(embedded_utterance, batch_first=True)
            embedded_schema = pad_sequence(embedded_schema, batch_first=True)
            # according to length of segment to identify which one is utterance/schema
            encoder_mask = pad_sequence(encoder_mask, batch_first=True)
            encoder_schema_mask = pad_sequence(encoder_schema_mask, batch_first=True)

            if self.use_schema_encoder:
                # resize schema and others
                embedded_schema = embedded_schema.view(batch_size * max_ent_size, max_ent_token_size, -1)
                expand_encoder_schema_mask = encoder_schema_mask.view(batch_size * max_ent_size, max_ent_token_size)

                # get the results, note the result is actually the final result of every column
                encoding_schema = self.schema_encoder.forward(embedded_schema, expand_encoder_schema_mask)
                encoding_schema = encoding_schema.view(batch_size, max_ent_size, -1)

                # batch_size x schema_size x utt_len
                linking_scores = embedded_utterance.new_zeros((batch_size, max_ent_size, utt_len))
                for ind in range(batch_size):
                    # schema_len x embedding_dim
                    batch_encoding_schema = encoding_schema[ind]
                    # chunk_size x utt_len x embedding_dim
                    batch_chunk_embedded_utt = chunk_embedded_utterance[ind]
                    batch_chunk_size = batch_chunk_embedded_utt.size(0)
                    # chunk_size x schema_len x embedding_dim
                    expand_encoding_schema = batch_encoding_schema.view(1, max_ent_size, -1)
                    expand_encoding_schema = expand_encoding_schema.expand(batch_chunk_size, max_ent_size, -1)
                    # chunk_size * max_ent_size x utt_len
                    batch_score = torch.bmm(expand_encoding_schema,
                                            torch.transpose(batch_chunk_embedded_utt, 1, 2)) / self.scale
                    # take the corresponding chunk score as the expected score
                    select_chunk = schema_local_position[ind].unsqueeze(dim=1).unsqueeze(dim=0)
                    select_chunk = select_chunk.expand(-1, -1, utt_len)
                    chunk_score = batch_score.gather(dim=0, index=select_chunk)
                    linking_scores[ind] = chunk_score[0]
            else:
                # encode table & column
                encoding_schema = embedded_schema.view(batch_size,
                                                       max_ent_size * max_ent_token_size,
                                                       self.embedding_dim)
                question_entity_similarity = torch.bmm(encoding_schema,
                                                       torch.transpose(embedded_utterance, 1, 2)) / self.scale

                # eps for nan loss
                encoder_sum = encoder_schema_mask.view(batch_size * max_ent_size,
                                                       max_ent_token_size).sum(dim=1).float() + 1e-2
                encoding_schema = encoding_schema.view(batch_size * max_ent_size,
                                                       max_ent_token_size, self.embedding_dim).sum(dim=1)
                encoding_schema = encoding_schema / encoder_sum.unsqueeze(dim=1).expand_as(encoding_schema)
                encoding_schema = encoding_schema.view(batch_size, max_ent_size, self.embedding_dim)

                # batch_size x expect_ent_size x col_token_size x utt_token_size
                question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                             max_ent_size,
                                                                             max_ent_token_size,
                                                                             -1)
                # batch_size x expect_ent_size x utt_token_size
                question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)
                linking_scores = question_entity_similarity_max_score

            if self.use_feature_score:
                linking_features = schema['linking']
                feature_size = linking_features.size(-1)
                # disable linking features of masked entity (non negative ones)
                linking_features = linking_features.view(batch_size, max_ent_size, -1, feature_size)
                # masking those unselected entities in this batch
                for ind in range(batch_size):
                    select_entity = select_entity_list[ind]
                    real_ent_size = real_ent_size_list[ind]
                    for ent_ind in range(real_ent_size):
                        if ent_ind not in select_entity:
                            linking_features[ind, ent_ind].fill_(0.0)

                # batch_size x expect_ent_size x utt_token_size
                feature_scores = self.linking_layer.forward(linking_features).squeeze(3)
                linking_scores = linking_scores + feature_scores

            parser_input = embedded_utterance

            if self.use_linking_embedding:
                entity_size = self.num_entity_types

                # batch_size x expect_ent_size x entity_size (10 now)
                entity_type_mat = torch.zeros((batch_size, max_ent_size, entity_size), dtype=torch.float32,
                                              device=device)

                # create one hot vector
                expand_entity_type = entity_type.unsqueeze(dim=2)
                entity_type_mat.scatter_(dim=2, index=expand_entity_type, value=1)

                # add 1e-8 as epsilon
                entity_type_mat = entity_type_mat + 1e-8

                # batch_size x utt_token_size x entity_size
                linking_probabilities = self._get_linking_probabilities(linking_scores.transpose(1, 2),
                                                                        entity_type_mat)

                linking_probabilities = encoder_mask.unsqueeze(dim=-1).repeat(1, 1,
                                                                              entity_size).float() * linking_probabilities

                # batch_size x entity_size x entity_embedding_size
                entity_ids = torch.arange(0, entity_size, 1, dtype=torch.long, device=device).unsqueeze(dim=0). \
                    repeat(batch_size, 1)
                entity_type_embeddings = self.link_entity_type_embedder.forward(entity_ids)
                entity_type_embeddings = torch.tanh(entity_type_embeddings)

                # calculate the weighted entity embeddings
                # batch_size x utt_token_size x entity_embedding_size
                linking_embedding = torch.bmm(linking_probabilities, entity_type_embeddings)
                parser_input = parser_input + linking_embedding
        else:
            raise Exception("DO NOT SUPPORT BERT MODE :{}".format(self.bert_mode))

        return parser_input, encoder_mask, linking_scores, encoding_schema

    def _init_grammar_state(self,
                            encoder_input: torch.FloatTensor,
                            encoder_mask: torch.LongTensor,
                            batch_size: int,
                            worlds: List[KBWorld],
                            semantic_cons: List[Dict[str, List]],
                            anchor_cons: List[Dict[str, Dict]],
                            literal_relation: List[Set[str]],
                            linking_scores: torch.FloatTensor,
                            valid_actions: List[List[ProductionRule]],
                            entity_type: torch.LongTensor,
                            encoding_schema: torch.FloatTensor):

        batch_size = batch_size

        # specific devices
        device = encoder_mask.device

        # encode and output encoder memory
        encoder_input = self.var_dropout(encoder_input)

        # an unified process to handle bert/non-bert embedding as input
        _, sequence_len, embedding_dim = encoder_input.size()

        utt_encoder_outputs = self.text_encoder(encoder_input, encoder_mask)
        utt_encoder_outputs = self.var_dropout(utt_encoder_outputs)
        # This will be our initial hidden state and memory cell for the decoder LSTM.

        # TODO: in the original mask, it will cause into fetching nothing because there may be an empty sentence.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs=utt_encoder_outputs,
                                                             mask=encoder_mask.bool(),
                                                             bidirectional=self.text_encoder.is_bidirectional())

        memory_cell = utt_encoder_outputs.new_zeros(batch_size, self.encoder_output_dim)

        initial_score = torch.zeros(batch_size, device=device, dtype=torch.float32)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [utt_encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [encoder_mask[i] for i in range(batch_size)]

        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            valid_actions[i],
                                                            linking_scores[i],
                                                            entity_type[i],
                                                            encoding_schema[i])
                                 for i in range(batch_size)]

        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self.first_action_embedding,
                                                 self.first_attended_output,
                                                 encoder_output_list,
                                                 utterance_mask_list))

        # initialize constrain state
        initial_condition_state = []
        for i in range(batch_size):
            if semantic_cons is None or semantic_cons[i] is None:
                type_constraint = {}
            else:
                type_constraint = semantic_cons[i]
            if anchor_cons is not None:
                anchor_constraint = anchor_cons[i]
            else:
                anchor_constraint = {}
            initial_condition_state.append(
                SExpressionState([a[0] for a in valid_actions[i]],
                                 worlds[i],
                                 worlds[i].language.is_nonterminal,
                                 relation_to_argument=type_constraint,
                                 anchor_to_argument=anchor_constraint,
                                 # always take the first element
                                 literal_relation=literal_relation[0] if literal_relation else None,
                                 model_in_training=self.training,
                                 enabled_type=self.use_type_checking,
                                 enabled_anchor=self.use_entity_anchor,
                                 enabled_virtual=self.use_virtual_forward,
                                 enabled_runtime_prune=self.use_runtime_prune))

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sexpression_state=initial_condition_state,
                                          possible_actions=valid_actions)
        return initial_state

    def _create_grammar_state(self,
                              world: KBWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              entity_types: torch.LongTensor,
                              encoding_schema: torch.Tensor) -> 'GrammarStatelet':
        """
        Construct initial grammar state let for decoding constraints
        :param world: ``SparcWorld``
        :param possible_actions: ``List[CopyProductionRule]``, tracking the all possible actions under current state
        this rule is different from the one in allennlp as it is support `is_copy` attribute
        :param linking_scores: ``torch.Tensor``, the linking score between every query token and each entity type
        :param entity_types: ``torch.Tensor``, the entity type of each schema in database
        :param encoding_schema: ``torch.Tensor``, the entity representation
        :return:
        """
        # map action into ind
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        translated_valid_actions = {}
        device = linking_scores.device

        # fake an empty Statement because there must be valid actions
        if world is None:
            translated_valid_actions['@start@'] = {}
            # assign to take away to keep consistent

            return GrammarStatelet(['@start@'],
                                   translated_valid_actions,
                                   # callback function
                                   lambda x: True)

        valid_actions = world.valid_actions
        action_to_entity = world.get_action_entity_mapping()

        for key, action_strings in valid_actions.items():
            # allocate dictionary
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action_map[action_string] for action_string in action_strings]

            ProductionTuple = namedtuple('ProductionTuple', ('rule', 'is_global', 'tensor', 'nonterminal'))
            # named tuple for better reading
            production_rule_arrays = [(ProductionTuple(*possible_actions[index]), index) for index in action_indices]

            # split rules into two category
            global_actions = []
            linked_actions = []

            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array.is_global:
                    global_actions.append((production_rule_array.tensor, action_index))
                else:
                    linked_actions.append((production_rule_array.rule, action_index))

            if global_actions:
                action_tensors, action_ids = zip(*global_actions)
                action_tensor = torch.cat(action_tensors, dim=0).long()

                # batch_size x embedding_size
                action_input_embedding = self.action_embedder.forward(action_tensor)
                action_output_embedding = self.output_action_embedder.forward(action_tensor)
                translated_valid_actions[key]['global'] = (action_input_embedding,
                                                           action_output_embedding,
                                                           list(action_ids))

            if linked_actions:
                # TODO: how to handle the embedding of *
                action_rules, action_ids = zip(*linked_actions)
                related_entity_ids = [action_to_entity[rule] for rule in action_rules]

                # assert related entity ids does not contain -1
                assert -1 not in related_entity_ids
                entity_linking_scores = linking_scores[related_entity_ids]

                if self.use_schema_as_input:
                    linked_action_embeddings = self.schema_to_action(encoding_schema[related_entity_ids])
                else:
                    entity_type_tensor = entity_types[related_entity_ids]
                    linked_action_embeddings = (self.output_entity_type_embedder(entity_type_tensor)
                                                .to(entity_types.device)
                                                .float())

                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                           linked_action_embeddings,
                                                           list(action_ids))

        return GrammarStatelet(['@start@'],
                               translated_valid_actions,
                               # callback function
                               world.language.is_nonterminal)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        iid_em = self._iid_split_accuracy.get_metric(reset)
        com_em = self._com_split_accuracy.get_metric(reset)
        zero_em = self._zero_split_accuracy.get_metric(reset)
        avg_em = self._avg_accuracy.get_metric(reset)
        iid_cover = self._cand_iid_accuracy.get_metric(reset)
        com_cover = self._cand_com_accuracy.get_metric(reset)
        zero_cover = self._cand_zero_accuracy.get_metric(reset)

        return_metrics = {
            "avg_exact_match": avg_em,
            "_iid_exact_match": iid_em,
            "com_exact_match": com_em,
            "zero_exact_match": zero_em,
            "_iid_recall": iid_cover,
            "_com_recall": com_cover,
            "_zero_recall": zero_cover
        }

        if self.evaluate_f1:
            iid_f1 = self._iid_split_f1.get_metric(reset)
            com_f1 = self._com_split_f1.get_metric(reset)
            zero_f1 = self._zero_split_f1.get_metric(reset)
            avg_f1 = self._avg_f1.get_metric(reset)

            return_metrics.update({
                "avg_f1": avg_f1,
                "_iid_f1": iid_f1,
                "_com_f1": com_f1,
                "_zero_f1": zero_f1
            })

        if self.evaluate_hits1:
            avg_hits1 = self._avg_hits1.get_metric(reset)
            return_metrics.update({
                "avg_hits1": avg_hits1
            })

        return return_metrics

    @staticmethod
    def _get_linking_probabilities(linking_scores: torch.Tensor,
                                   entity_type_mat: torch.LongTensor) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size, utt_token_size, col_size).
        entity_type_mat : ``torch.LongTensor``
            Has shape (batch_size, col_size, entity_size)
        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size, utt_token_size, entity_size)``.
            Contains all the probabilities of entity types given an utterance word
        """
        # normalize entity type mat into probability
        entity_type_base = entity_type_mat.sum(dim=2, keepdim=True).expand_as(entity_type_mat)
        # divide and get the probability, batch_size x col_size x entity_size
        entity_type_prob = entity_type_mat / entity_type_base
        # bmm and get the result, batch_size x utt_token_size x entity_size
        type_linking_score = torch.bmm(linking_scores, entity_type_prob)
        # normalize on entity dimension
        type_linking_prob = torch.softmax(type_linking_score, dim=2)

        return type_linking_prob
