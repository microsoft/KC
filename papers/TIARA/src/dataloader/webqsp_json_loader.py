# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re

from dataloader.json_loader import JsonLoader


class WebQSPJsonLoader(JsonLoader):
    data: list
    len: int
    idf_denominator: dict  # for inverse document frequency
    typo_dict: dict
    elq_cache: dict

    def __init__(self, file_path: str, typo_dict_file_path=None):
        self.file_path = file_path
        with open(file_path, 'r', encoding='UTF-8') as file:
            self.data = json.load(file)
            self.len = len(self.data)

        self.typo_dict = dict()
        if typo_dict_file_path is not None:
            with open(typo_dict_file_path, 'r', encoding='UTF-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    line_split = line.split('\t')
                    key = line_split[0]
                    if key not in self.typo_dict:
                        self.typo_dict[key] = line_split[1]

    def get_len(self):
        return self.len

    def get_question_id_by_idx(self, idx) -> str:
        return self.data[idx]['QuestionId']

    def get_question_by_idx(self, idx) -> str:
        ques = self.data[idx]['ProcessedQuestion']
        for key in self.typo_dict.keys():
            if (key + ' ') in ques or (' ' + key) in ques:
                ques = ques.replace(key, self.typo_dict[key])
        return ques

    def get_masked_question_with_token_by_idx(self, idx) -> str:
        question = self.get_question_by_idx(idx)
        for mention in self.get_potential_topic_entity_mention_by_idx(idx):
            if mention is None:
                continue
            question = question.replace(mention.lower(), "<e>")
        return question

    def get_masked_question_with_type_by_idx(self, idx) -> str:
        question = self.get_question_by_idx(idx)
        for parse in self.data[idx]['Parses']:
            if len(parse['InferentialChain']) == 0:
                continue
            entity_type = parse['InferentialChain'][0].split('.')[1].replace('_', ' ')
            mention = parse['PotentialTopicEntityMention']
            if mention is None:
                continue
            question = question.replace(mention.lower(), entity_type)
        return question

    def get_entity_masked_questions(self):
        res = []
        for idx in range(0, self.len):
            res.append(self.get_masked_question_with_token_by_idx(idx))
        return res

    def get_potential_topic_entity_mention_by_idx(self, idx) -> list:
        res = []
        for parse in self.data[idx]['Parses']:
            if parse['PotentialTopicEntityMention'] is not None:
                res.append(parse['PotentialTopicEntityMention'])
        return res

    def get_golden_entity_by_idx(self, idx, return_dict_list=False):
        name_list = []
        mid_list = []
        for parse in self.data[idx]['Parses']:
            name_list.append(parse['TopicEntityName'])
            mid_list.append(parse['TopicEntityMid'])
        assert len(name_list) == len(mid_list)

        if return_dict_list:
            entities = []
            for i in range(0, len(name_list)):
                if name_list[i] is not None and mid_list[i] is not None:
                    entities.append({'friendly_name': name_list[i], 'id': mid_list[i]})
            return entities
        return name_list, mid_list

    def get_golden_entity_mid_by_idx(self, idx) -> list:
        res = []
        for parse in self.data[idx]['Parses']:
            res.append(parse['TopicEntityMid'])
        return res

    def get_golden_entity_mid_in_first_parse_by_idx(self, idx) -> list:
        return self.data[idx]['Parses'][0]['TopicEntityMid']

    def get_golden_entity_name_by_idx(self, idx) -> list:
        res = []
        for parse in self.data[idx]['Parses']:
            res.append(parse['TopicEntityName'])
        return res

    def get_golden_relation_by_idx(self, idx):
        res = []
        for parse in self.data[idx]['Parses']:
            if parse['InferentialChain'] is not None:
                res += parse['InferentialChain']
        return list(set(res))

    def get_golden_relation_by_masked_question(self, masked_question: str) -> list:
        res = []
        for idx in range(0, self.len):
            if masked_question == self.get_masked_question_with_token_by_idx(idx):
                res += self.get_golden_relation_by_idx(idx)
        return res

    def get_golden_relations(self):
        return super().get_golden_relations()

    def get_golden_inferential_chains(self):
        res = set()
        for idx in range(0, self.len):
            for parse in self.data[idx]['Parses']:
                res.add(str(parse['InferentialChain']))
        return res

    def get_golden_inferential_chain_in_parses_by_idx(self, idx):
        res = []
        for parse in self.data[idx]['Parses']:
            res.append(str(parse['InferentialChain']))
        return res

    def get_inferential_chain_question_dict(self):
        res = dict()
        for idx in range(0, self.len):
            masked_question = self.get_masked_question_with_token_by_idx(idx)
            for parse in self.data[idx]['Parses']:
                if parse['InferentialChain'] is None:
                    continue
                ic_str = str(parse['InferentialChain'])
                if ic_str not in res:
                    res[ic_str] = set()
                res[ic_str].add(masked_question)
        return res

    def get_masked_question_set(self):
        res = set()
        for idx in range(0, self.len):
            res.add(self.get_masked_question_with_token_by_idx(idx))
        return res

    def get_ans_list_by_idx(self, idx) -> list:
        """
        Get answers of a question.
        Args:
            idx: question id of training/test set

        Returns:
            An answer list
        """
        res = []
        for parse in self.data[idx]['Parses']:
            res += parse['Answers']
        return res

    def get_ans_arg_list_by_idx(self, idx) -> list:
        res = []
        for ans in self.get_ans_list_by_idx(idx):
            res.append(ans['AnswerArgument'])
        return list(set(res))

    def get_ans_arg_list_in_parses_by_idx(self, idx) -> list:
        res = []
        for parse in self.data[idx]['Parses']:
            parse_ans = []
            for ans in parse['Answers']:
                parse_ans.append(ans['AnswerArgument'])
            res.append(parse_ans)  # each element for one parse
        return res

    def get_ans_name_list_by_idx(self, idx) -> list:
        res = []
        for ans in self.get_ans_list_by_idx(idx):
            if ans['AnswerType'] == 'Value':
                res.append(ans['AnswerArgument'])
            elif ans['AnswerType'] == 'Entity':
                res.append(ans['EntityName'])
        return res

    def get_ans_name_and_mid_list_by_idx(self, idx) -> list:
        res = []
        for ans in self.get_ans_list_by_idx(idx):
            if ans['AnswerType'] == 'Value':
                res.append(ans['AnswerArgument'])
            elif ans['AnswerType'] == 'Entity':
                res.append((ans['EntityName'], ans['AnswerArgument']))
        return res

    def get_sparql_by_idx(self, idx):
        return self.data[idx]['Parses'][0]['Sparql']

    def get_sparql_list(self):
        res = []
        for idx in range(0, self.len):
            sparql = self.get_sparql_by_idx(idx)
            res.append(sparql)
        return res

    def get_s_expression_by_idx(self, idx):
        return self.data[idx]['Parses'][0]['SExpr']

    def get_s_expression_parses_by_idx(self, idx):
        res = []
        for parse in self.data[idx]['Parses']:
            if 'SExpr' in parse and parse['SExpr'] is not None and parse['SExpr'] != 'null':
                res.append(parse['SExpr'])
        return res

    def get_constraints_by_idx(self, idx: int):
        res = []
        for parse in self.data[idx]['Parses']:
            res.append(parse['Constraints'])
        return res

    def get_orders_by_idx(self, idx: int):
        res = []
        for parse in self.data[idx]['Parses']:
            res.append(parse['Order'])
        return res

    def get_constraints(self):
        res = []
        for idx in range(0, self.len):
            for parse in self.data[idx]['Parses']:
                res.append(parse['Constraints'])
        return res

    def get_constraint_entity_by_idx(self, idx: int):
        res = set()
        constraints = self.get_constraints_by_idx(idx)
        for constraint in constraints:
            for c in constraint:
                res.add(c['Argument'])
        return list(res)

    def get_constraint_predicate_set(self):
        res = set()
        for constraint in self.get_constraints():
            for d in constraint:
                res.add(d['NodePredicate'])
        return res

    def get_constraint_predicate_by_idx(self, idx: int):
        res = set()
        constraints = self.get_constraints_by_idx(idx)
        for constraint in constraints:
            for c in constraint:
                res.add(c['NodePredicate'])
        return list(res)

    def get_order_predicate_by_idx(self, idx: int):
        res = set()
        orders = self.get_orders_by_idx(idx)
        for order in orders:
            if order is None or order['NodePredicate'] is None:
                continue
            res.add(order['NodePredicate'])
        return list(res)

    def get_question_predicates_by_idx(self, idx):
        res = set()
        res.update(self.get_golden_relation_by_idx(idx))
        res.update(self.get_constraint_predicate_by_idx(idx))
        res.update(self.get_order_predicate_by_idx(idx))
        return res

    def get_topic_entity_dict(self):
        res = dict()
        for idx in range(0, self.len):
            for parse in self.data[idx]['Parses']:
                potential_topic_entity_mention = parse['PotentialTopicEntityMention']
                if potential_topic_entity_mention is None or len(potential_topic_entity_mention) == 0:
                    continue
                res[parse['PotentialTopicEntityMention']] = (parse['TopicEntityMid'], parse['TopicEntityName'])
        return res

    def get_topic_entity_mid_label_mention_by_idx(self, idx):
        label_mention_list = []  # each element is a label and a mention for one topic entity
        mid_list = []
        for parse in self.data[idx]['Parses']:
            label = parse['TopicEntityName']
            mention = parse['PotentialTopicEntityMention']
            mid = parse['TopicEntityMid']

            label_mention_list.append([label, mention])
            mid_list.append(mid)
        return label_mention_list, mid_list

    def get_constraint_entity_dict(self):
        res = dict()
        for idx in range(0, self.len):
            constraints = self.get_constraints_by_idx(idx)
            for constraint in constraints:
                for c in constraint:
                    if c['EntityName'] is not None and c['EntityName'] != '':
                        res[c['EntityName']] = c['Argument']
        return res

    def get_ns_entities_relations(self):
        res = set()
        for idx in range(0, self.len):
            sparql = self.get_sparql_by_idx(idx)
            entities_relations = re.findall('ns:[a-zA-Z0-9_\.]+', sparql)
            for ns in entities_relations:
                res.add(ns)
        return res

    def get_question_with_mid_by_idx(self, idx):
        question = self.get_question_by_idx(idx)
        ner = self.get_potential_topic_entity_mention_by_idx(idx)
        topic_mid = self.get_golden_entity_mid_in_first_parse_by_idx(idx)
        if ner is not None:
            ner_start = question.find(ner[0])
            ner_end = ner_start + len(ner[0])
            question = question[: ner_end] + ' ' + 'ns:' + topic_mid + question[ner_end:]
        return question
