# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import re
from collections import defaultdict
from typing import List, Optional

import networkx as nx
import torch
from allennlp.training.metrics import Metric

logger = logging.getLogger(__name__)


def clean_answer_tags(answer):
    if "^^<http://www.w3.org/2001/XMLSchema#" in answer:
        answer = re.sub("XMLSchema#\w+>", "", answer.replace("^^<http://www.w3.org/2001/", ""))
        answer = answer.strip("\"")
    if re.fullmatch(r"\d{4}Z", answer) or re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", answer):
        answer = answer.rstrip("Z")
    return answer


@Metric.register("set_f1")
class SetF1Measure(Metric):

    def __init__(self):
        super(SetF1Measure, self).__init__()
        self.total_f1_sum = 0.0
        self.total_count = 0

    def __call__(self, predict_answers: List[List], golden_answers: List[List],
                 mask: Optional[torch.BoolTensor] = None):
        if mask is None:
            mask = [1] * len(predict_answers)
        # clean prediction since it may contain XML tag (not in groundtruth)
        clean_prediction = [list(map(clean_answer_tags, example)) for example in predict_answers]
        for predict_answer, gold_answer, count_mask in zip(clean_prediction, golden_answers, mask):
            if count_mask:
                try:
                    predict_answer = set(predict_answer)
                    gold_answer = set(gold_answer)
                    if len(predict_answer) == 0:
                        precision = 0.0
                    else:
                        precision = len(predict_answer.intersection(gold_answer)) / len(predict_answer)
                    recall = len(predict_answer.intersection(gold_answer)) / len(gold_answer)
                    if recall != 0.0 and precision != 0.0:
                        self.total_f1_sum += (2 * recall * precision / (recall + precision))
                    else:
                        self.total_f1_sum += 0.0

                except Exception as e:
                    print(e)
                    self.total_f1_sum += 0.0
                self.total_count += 1

    def reset(self) -> None:
        self.total_f1_sum = 0
        self.total_count = 0

    def get_metric(self, reset: bool) -> float:
        metric = self.total_f1_sum / self.total_count if self.total_count != 0 else 0.0
        if reset: self.reset()
        return metric

    def equals(self, predict_answer, golden_answer):
        clean_predict = list(map(clean_answer_tags, predict_answer))
        return set(clean_predict) == set(golden_answer)


@Metric.register("set_hit1")
class SetHit1Measure(Metric):

    def __init__(self):
        super(SetHit1Measure, self).__init__()
        self.total_count = 0
        self.total_hit_sum = 0.0

    def __call__(self, predict_answers: List[List], golden_answers: List[List],
                 mask: Optional[torch.BoolTensor] = None):
        if mask is None:
            mask = [1] * len(predict_answers)
        # clean prediction since it may contain XML tag (not in groundtruth)
        clean_prediction = [list(map(clean_answer_tags, example)) for example in predict_answers]
        for predict_answer, gold_answer, count_mask in zip(clean_prediction, golden_answers, mask):
            if count_mask:
                try:
                    predict_answer = set(predict_answer)
                    gold_answer = set(gold_answer)
                    if len(predict_answer):
                        one_ans = random.sample(predict_answer, 1)[0]
                        hits1 = int(one_ans in gold_answer)
                    else:
                        hits1 = 0.0
                    self.total_hit_sum += hits1
                except Exception as e:
                    print(e)
                    self.total_hit_sum += 0.0
                self.total_count += 1

    def reset(self) -> None:
        self.total_hit_sum = 0
        self.total_count = 0

    def get_metric(self, reset: bool) -> float:
        metric = self.total_hit_sum / self.total_count if self.total_count != 0 else 0.0
        if reset: self.reset()
        return metric


@Metric.register("graph_match")
class GraphMatchMeasure(Metric):

    def __init__(self, fb_roles_file,
                 fb_types_file,
                 reverse_properties_file):
        super(GraphMatchMeasure, self).__init__()
        if fb_roles_file is not None:
            reverse_properties, relation_dr, relations, \
            upper_types, types = GraphMatchMeasure._process_ontology(fb_roles_file,
                                                                     fb_types_file,
                                                                     reverse_properties_file)
            matcher = SemanticMatcher(reverse_properties, relation_dr,
                                      relations, upper_types, types)
            # predict_lf, golden_lf
            self.matches = matcher.same_logical_form
        else:
            # exact match
            logger.warning("You're using Exact Match instead of Graph Exact Match "
                           "since you pass by an empty fb_roles_file and other files.")
            self.matches = lambda x, y: x == y
        self.total_match_sum = 0.0
        self.total_count = 0

    def __call__(self, predict_lfs: List[str], golden_lfs: List[str],
                 mask: Optional[torch.BoolTensor] = None):
        if mask is None:
            mask = [1] * len(predict_lfs)
        for predict_lf, golden_lf, count_mask in zip(predict_lfs, golden_lfs, mask):
            if count_mask:
                self.total_count += 1
                if self.matches(predict_lf, golden_lf):
                    self.total_match_sum += 1

    def get_metric(self, reset: bool) -> float:
        metric = self.total_match_sum / self.total_count if self.total_count != 0 else 0.0
        if reset: self.reset()
        return metric

    def reset(self) -> None:
        self.total_match_sum = 0.0
        self.total_count = 0

    def equals(self, predict_sexpression, golden_sexpression):
        return self.matches(predict_sexpression, golden_sexpression)

    @staticmethod
    def _process_ontology(fb_roles_file, fb_types_file, reverse_properties_file):
        reverse_properties = {}
        with open(reverse_properties_file, 'r') as f:
            for line in f:
                reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

        with open(fb_roles_file, 'r') as f:
            content = f.readlines()

        relation_dr = {}
        relations = set()
        for line in content:
            fields = line.split()
            relation_dr[fields[1]] = (fields[0], fields[2])
            relations.add(fields[1])

        with open(fb_types_file, 'r') as f:
            content = f.readlines()

        upper_types = defaultdict(lambda: set())

        types = set()
        for line in content:
            fields = line.split()
            upper_types[fields[0]].add(fields[2])
            types.add(fields[0])
            types.add(fields[2])

        return reverse_properties, relation_dr, relations, upper_types, types


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
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


function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}


class SemanticMatcher:
    def __init__(self, reverse_properties, relation_dr, relations, upper_types, types):
        self.reverse_properties = reverse_properties
        self.relation_dr = relation_dr
        self.relations = relations
        self.upper_types = upper_types
        self.types = types

    def same_logical_form(self, form1, form2):
        if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
            return False
        try:
            G1 = self.logical_form_to_graph(lisp_to_nested_expression(form1))
        except Exception:
            return False
        try:
            G2 = self.logical_form_to_graph(lisp_to_nested_expression(form2))
        except Exception:
            return False

        def node_match(n1, n2):
            if n1['id'] == n2['id'] and n1['type'] == n2['type']:
                func1 = n1.pop('function', 'none')
                func2 = n2.pop('function', 'none')
                tc1 = n1.pop('tc', 'none')
                tc2 = n2.pop('tc', 'none')

                if func1 == func2 and tc1 == tc2:
                    return True
                else:
                    return False
                # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
                #     return True
                # elif 'function' not in n1 and 'function' not in n2:
                #     return True
                # else:
                #     return False
            else:
                return False

        def multi_edge_match(e1, e2):
            if len(e1) != len(e2):
                return False
            values1 = []
            values2 = []
            for v in e1.values():
                values1.append(v['relation'])
            for v in e2.values():
                values2.append(v['relation'])
            return sorted(values1) == sorted(values2)

        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)

    def get_symbol_type(self, symbol: str) -> int:
        if symbol.__contains__('^^'):  # literals are expected to be appended with data types
            return 2
        elif symbol in self.types:
            return 3
        elif symbol in self.relations:
            return 4
        else:
            return 1

    def logical_form_to_graph(self, expression: List) -> nx.MultiGraph:
        # TODO: merge two entity node with same id. But there is no such need for
        # the second version of graphquestions
        G = self._get_graph(expression)
        G.nodes[len(G.nodes())]['question_node'] = 1
        return G

    def _get_graph(self,
                   expression: List) -> nx.MultiGraph:
        # The id of question node is always the same as the size of the graph
        if isinstance(expression, str):
            G = nx.MultiDiGraph()
            if self.get_symbol_type(expression) == 1:
                G.add_node(1, id=expression, type='entity')
            elif self.get_symbol_type(expression) == 2:
                G.add_node(1, id=expression, type='literal')
            elif self.get_symbol_type(expression) == 3:
                G.add_node(1, id=expression, type='class')
                # G.add_node(1, id="common.topic", type='class')
            elif self.get_symbol_type(expression) == 4:  # relation or attribute
                domain, rang = self.relation_dr[expression]
                G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
                G.add_node(2, id=domain, type='class')
                G.add_edge(2, 1, relation=expression)

                if expression in self.reverse_properties:  # take care of reverse properties
                    G.add_edge(1, 2, relation=self.reverse_properties[expression])

            return G

        if expression[0] == 'R':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            mapping = {}
            for n in G.nodes():
                mapping[n] = size - n + 1
            G = nx.relabel_nodes(G, mapping)
            return G

        elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
            G1 = self._get_graph(expression=expression[1])
            G2 = self._get_graph(expression=expression[2])
            size = len(G2.nodes())
            qn_id = size
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in self.upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
                # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G = nx.compose(G1, G2)

            if expression[0] != 'JOIN':
                G.nodes[1]['function'] = function_map[expression[0]]

            return G

        elif expression[0] == 'AND':
            G1 = self._get_graph(expression[1])
            G2 = self._get_graph(expression[2])

            size1 = len(G1.nodes())
            size2 = len(G2.nodes())
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
                # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
                # So here for the AND function we force it to choose the type explicitly provided in the logical form
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'COUNT':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['function'] = 'count'

            return G

        elif expression[0].__contains__('ARG'):
            G1 = self._get_graph(expression[1])
            size1 = len(G1.nodes())
            G2 = self._get_graph(expression[2])
            size2 = len(G2.nodes())
            # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
            G2.nodes[1]['id'] = 0
            G2.nodes[1]['type'] = 'literal'
            G2.nodes[1]['function'] = expression[0].lower()
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']

            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'TC':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['tc'] = (expression[2], expression[3])

            return G
