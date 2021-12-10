# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import pickle
import sys
import traceback
from typing import List, Dict, Iterable, Set, Optional

import dill
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, ArrayField, LabelField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Tokenizer
from allennlp.data.tokenizers.token import Token
from allennlp_semparse.fields import ProductionRuleField
from overrides import overrides

from converter import lisp_to_verify_sparql_xsl_only, lisp_to_sparql, webqsp_lisp_to_verify_sparql_xsl_only, webqsp_lisp_to_sparql
from kb_utils.kb_context import KBContext
from kb_utils.knowledge_field import GrailKnowledgeGraphField
from languages import SExpressionLanguage, WebQSPSExpressionLanguage
from utils import Class, Relation, UNK_CLASS, UNK_REL, UNK_ENT, EntityEncode, GeneralizeLevel, RegisterWorld
from worlds import GrailKBWorld, WebKBWorld

logger = logging.getLogger(__name__)


@DatasetReader.register("grail")
class GrailDatasetReader(DatasetReader):
    def __init__(self,
                 world: str,
                 lazy: bool = False,
                 use_bert: bool = True,
                 # shuffle the entity/relation to avoid data leak
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 # see utils.py EntityEncode
                 encode_method: str = "self",
                 semantic_constrained_file: str = None,
                 literal_relation_file: str = None,
                 truncate_utterance_len: int = 50,
                 maximum_eval_cand_len: int = 750,
                 loading_limit=-1,
                 **kwargs):
        super().__init__(lazy=lazy, **kwargs)

        if tokenizer is None:
            self._tokenizer = PretrainedTransformerTokenizer(
                model_name="bert-base-uncased",
                add_special_tokens=True,
                max_length=512
            )
        else:
            self._tokenizer = tokenizer
        if token_indexers is None:
            # specify BERT base as default option
            self._token_indexer = {
                "bert": PretrainedTransformerIndexer(
                    model_name="bert-base-uncased",
                    max_length=512
                )
            }
        else:
            self._token_indexer = token_indexers

        self._use_bert = use_bert
        self._loading_limit = loading_limit
        self._level_to_dataset = {
            "i.i.d.": GeneralizeLevel.iid,
            "compositional": GeneralizeLevel.com,
            "zero-shot": GeneralizeLevel.zero
        }

        self.semantic_constraint_file = semantic_constrained_file
        self.literal_relation_file = literal_relation_file
        self._truncate_utterance_len = truncate_utterance_len
        self._encode_method = encode_method
        assert encode_method in EntityEncode.values(), "You should specify one of encode method in the following: {}". \
            format(" ".join(EntityEncode.values()))

        # although we employ negative sampling
        self._maximum_eval_cand_len = maximum_eval_cand_len

        assert world in RegisterWorld.values()
        self._world = world

    def build_data_sample(self, ex: Dict, key: str, is_training: bool = False):
        question = ex["question"] if "question" in ex else ex["sentence"]
        question_id = ex["qid"]

        # key = "graph_query"
        answers = [ins_answer["answer_argument"] for ins_answer in ex["answer"]] if "answer" in ex else []
        entity_list = [Class(entity_id=ins_entity["id"],
                             entity_class=ins_entity["class"],
                             friendly_name=ins_entity["friendly_name"],
                             node_type=ins_entity["node_type"],
                             node_id=ins_entity["nid"],
                             all_entity_classes=ins_entity["classes"] if "classes" in ins_entity else None)
                       for ins_entity in ex[key]["nodes"]]
        relation_list = [Relation(relation_class=ins_relation["relation"],
                                  friendly_name=ins_relation["friendly_name"])
                         for ins_relation in ex[key]["edges"]]
        sexpression = ex["s_expression"] if "s_expression" in ex else ""
        sparql_query = ex["sparql_query"] if "sparql_query" in ex else ""
        level = 'i.i.d.' if 'level' not in ex else ex['level']

        if len(entity_list) == 0:
            # add UNK to avoid
            entity_list.append(UNK_CLASS)
            entity_list.append(UNK_ENT)
        elif len(entity_list) >= self._maximum_eval_cand_len:
            entity_list = entity_list[:self._maximum_eval_cand_len]

        if len(relation_list) == 0:
            relation_list.append(UNK_REL)
        elif len(relation_list) >= self._maximum_eval_cand_len:
            relation_list = relation_list[:self._maximum_eval_cand_len]

        if is_training:
            # ground list in training to be used in negative sampling
            ground_entity_list = [Class(entity_id=ins_entity["id"],
                                        entity_class=ins_entity["class"],
                                        friendly_name=ins_entity["friendly_name"],
                                        node_type=ins_entity["node_type"],
                                        node_id=ins_entity["nid"])
                                  for ins_entity in ex[key]["nodes"]]
            for entity in ground_entity_list:
                if entity in entity_list:
                    entity_list.remove(entity)
            ground_entity_list = list(set(ground_entity_list))
            ground_relation_list = [Relation(relation_class=ins_relation["relation"],
                                             friendly_name=ins_relation["friendly_name"])
                                    for ins_relation in ex[key]["edges"]]
            for relation in ground_relation_list:
                if relation in relation_list:
                    relation_list.remove(relation)
            ground_relation_list = list(set(ground_relation_list))
        else:
            ground_entity_list = []
            ground_relation_list = []

        # remove duplicates
        # entity_list = list(set(entity_list))
        # relation_list = list(set(relation_list))

        # record information for debugging on development
        graph_query_info = ex['graph_query'] if 'graph_query' in ex else None
        entity_meta_info = ex['entity_meta_info'] if 'entity_meta_info' in ex else None
        # record anchor constraint information
        anchor_cons_info = ex['anchor_relations'] if 'anchor_relations' in ex else None

        return {
            "question": question,
            "question_id": question_id,
            "entity_list": entity_list,
            "relation_list": relation_list,
            "ground_entity_list": ground_entity_list,
            "ground_relation_list": ground_relation_list,
            "sexpression": sexpression,
            "sparql_query": sparql_query,
            "level": level,
            "answers": answers,
            "graph_query_info": graph_query_info,
            "anchor_cons_info": anchor_cons_info,
            "entity_meta_info": entity_meta_info
        }

    @overrides
    def _read(self, file_path: str):
        if not file_path.endswith('.json'):
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

        is_training = "train" in file_path
        if not is_training and self.semantic_constraint_file:
            constraint_f = open(self.semantic_constraint_file, "rb")
            # head: relation: tail
            all_relation_dict: Optional[Dict[str, List]] = pickle.load(constraint_f)
        else:
            all_relation_dict = None

        if not is_training and self.literal_relation_file:
            literal_f = open(self.literal_relation_file, "rb")
            all_litearl_relation: Optional[Set] = pickle.load(literal_f)
        else:
            all_litearl_relation = None

        cnt = 0
        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for total_cnt, ex in enumerate(json_obj):
                # early stop for debugging
                if self._loading_limit == cnt:
                    break

                key = "graph_query" if "candidate_query" not in ex else "candidate_query"

                if not is_training and self.semantic_constraint_file:
                    candidate_relations = set([node["relation"] for node in ex[key]["edges"]])
                    relation_dict = {k: v for k, v in all_relation_dict.items() if k in candidate_relations}
                else:
                    relation_dict = None

                # TODO: now use the groundtruth as candidates, which should be prevented by shuang's candidates
                try:
                    data_sample = self.build_data_sample(ex, key, is_training)
                    ins = self.text_to_instance(semantic_cons=relation_dict,
                                                literal_relation=all_litearl_relation,
                                                is_training=is_training,
                                                **data_sample)
                except Exception as e:
                    print(f'Error in qid: {ex["qid"]}')
                    exec_info = sys.exc_info()
                    traceback.print_exception(*exec_info)

                if ins is not None:
                    cnt += 1

                if ins is not None:
                    yield ins

    @overrides
    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        print('read instance from', cache_filename)
        with open(cache_filename, 'rb') as cache_file:
            instances = dill.load(cache_file)
            for instance in instances:
                yield instance

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        print('write instance to', cache_filename)
        with open(cache_filename, 'wb') as cache_file:
            dill.dump(instances, cache_file)

    def text_to_instance(self,
                         question: str,
                         question_id: str,
                         # TODO: provided by Shuang
                         entity_list: List[Class],
                         relation_list: List[Relation],
                         ground_entity_list: List[Class],
                         ground_relation_list: List[Relation],
                         sexpression: str = None,
                         # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs ...
                         sparql_query: str = None,
                         level: str = 'i.i.d.',
                         # TODO: does it have any usage
                         semantic_cons: Dict = None,
                         literal_relation: set = None,
                         is_training: bool = False,
                         # useful for evaluation
                         # i.i.d, compositional, zero-shot
                         answers: List[str] = None,
                         graph_query_info: Dict = None,
                         entity_meta_info: Dict = None,
                         entity_offset_tokens: List = None,
                         anchor_cons_info: Dict = None):

        fields: Dict[str, Field] = {}

        tokenized_utterance = self._tokenizer.tokenize(question)
        tokenized_utterance = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
                               else Token(text=t.text, lemma_=t.text) for t in tokenized_utterance]
        tokenized_utterance = tokenized_utterance[:self._truncate_utterance_len]

        entity_list = ground_entity_list + entity_list
        relation_list = ground_relation_list + relation_list

        # order entity list and relation list
        # according to alphabet order to order the entity/relation
        entity_list.sort(key=lambda x: str(x))
        relation_list.sort(key=lambda x: str(x))

        kb_context = KBContext(tokenizer=self._tokenizer,
                               utterance=tokenized_utterance,
                               entity_list=entity_list,
                               relation_list=relation_list,
                               encode_method=self._encode_method)

        schema_field = GrailKnowledgeGraphField(kb_context.knowledge_graph,
                                                tokenized_utterance,
                                                self._token_indexer,
                                                entity_tokens=kb_context.entity_tokens,
                                                include_in_vocab=False,
                                                max_table_tokens=None)

        # build faked ones to enable inference on test cases
        if sexpression is None:
            # in test, we cannot obtain golden sexpression,
            # and we should build faked parsed sexpression
            sparql_query = ""
            select_entity = kb_context.entity_list[0].entity_id
            sexpression = f"(COUNT {select_entity})"

        if self._world == RegisterWorld.WebQSP:
            world = WebKBWorld(world_id=question_id,
                               kb_context=kb_context,
                               sparql_query=sparql_query,
                               sexpression=sexpression,
                               origin_utterance=question,
                               answers=answers,
                               level=level,
                               graph_query_info=graph_query_info,
                               entity_meta_info=entity_meta_info,
                               entity_offset_tokens=entity_offset_tokens,
                               language_class=WebQSPSExpressionLanguage,
                               verify_sparql_converter=webqsp_lisp_to_verify_sparql_xsl_only,
                               sparql_converter=webqsp_lisp_to_sparql)
        else:
            world = GrailKBWorld(world_id=question_id,
                                 kb_context=kb_context,
                                 sparql_query=sparql_query,
                                 sexpression=sexpression,
                                 origin_utterance=question,
                                 answers=answers,
                                 level=level,
                                 graph_query_info=graph_query_info,
                                 entity_meta_info=entity_meta_info,
                                 entity_offset_tokens=entity_offset_tokens,
                                 language_class=SExpressionLanguage,
                                 verify_sparql_converter=lisp_to_verify_sparql_xsl_only,
                                 sparql_converter=lisp_to_sparql)

        action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, rhs = production_rule.split(' -> ')
            field = ProductionRuleField(production_rule,
                                        world.language.is_global_rule(rhs),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        action_map = {action.rule: i  # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        if is_training or sexpression:
            if action_sequence:
                for production_rule in action_sequence:
                    index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
            else:
                index_fields = [IndexField(-1, valid_actions_field)]
            action_sequence_field = ListField(index_fields)
            fields["action_sequence"] = action_sequence_field

        if self._use_bert:
            # TODO: multiple chunk
            entity_boundary = [0, len(ground_entity_list)]
            relation_boundary = [kb_context.relation_start_index,
                                 kb_context.relation_start_index + len(ground_relation_list)]

            fields['ground_entity_start_end'] = ArrayField(np.array(entity_boundary, dtype=np.int))
            fields['ground_relation_start_end'] = ArrayField(np.array(relation_boundary, dtype=np.int))

        fields["utterance"] = TextField(tokenized_utterance,
                                        self._token_indexer)
        fields["dataset_type"] = LabelField(self._level_to_dataset[level], skip_indexing=True)
        fields["schema"] = schema_field
        fields["entity_type"] = ArrayField(world.index_entity_type())
        fields["valid_actions"] = valid_actions_field
        fields["worlds"] = MetadataField(world)

        if semantic_cons:
            # use pandas for quick indexing, only valid on development
            fields["semantic_cons"] = MetadataField(semantic_cons)

        if literal_relation:
            fields["literal_relation"] = MetadataField(literal_relation)

        if anchor_cons_info:
            fields["anchor_cons"] = MetadataField(anchor_cons_info)
        else:
            fields["anchor_cons"] = MetadataField({})

        return Instance(fields)
