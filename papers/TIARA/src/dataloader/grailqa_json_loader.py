# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re

from dataloader.json_loader import JsonLoader
from utils.domain_dict import fb_domain_dict


class GrailQAJsonLoader(JsonLoader):

    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='UTF-8') as file:
            f = json.load(file)
            self.data = f
            self.len = len(self.data)
            self.file_path = file_path

        self.question_id_to_idx_dict = dict()  # question_id: str -> idx: int
        self.build_question_id_to_idx_dict()

    def build_question_id_to_idx_dict(self):
        for idx in range(0, self.len):
            question_id = self.get_question_id_by_idx(idx)
            self.question_id_to_idx_dict[str(question_id)] = idx

    def get_idx_by_question_id(self, question_id):
        """
        Get the index in the json file by the question id
        :param question_id: the GrailQA question id
        :return: if the question is in the file, return the index, otherwise -1
        """
        question_id = str(question_id)
        if self.question_id_to_idx_dict is None or len(self.question_id_to_idx_dict) == 0:
            self.build_question_id_to_idx_dict()
        return self.question_id_to_idx_dict.get(question_id, -1)

    def get_sparql_by_idx(self, idx):
        return self.data[idx]['sparql_query']

    def get_s_expression_by_idx(self, idx):
        return self.data[idx]['s_expression']

    def get_function_by_idx(self, idx):
        return self.data[idx]['function']

    def get_domains_by_idx(self, idx):
        return self.data[idx]['domains']

    def get_level_by_idx(self, idx):
        if 'level' in self.data[idx]:
            return self.data[idx]['level']
        return 'n/a'

    def get_question_id_by_idx(self, idx, format='int'):
        qid = self.data[idx]['qid']
        if format == 'str':
            return str(qid)
        return qid

    def get_question_by_idx(self, idx):
        return self.data[idx]['question']

    def get_ans_list_by_idx(self, idx):
        return self.data[idx]['answer']

    def get_ans_arg_by_idx(self, idx):
        answers = self.data[idx]['answer']
        res = []
        for answer in answers:
            res.append(answer['answer_argument'])
        return res

    def get_len(self):
        return self.len

    def get_graph_query(self, idx):
        return self.data[idx]['graph_query']

    def print_sparqls(self):
        for idx in range(0, self.len):
            print(self.get_sparql_by_idx(idx))
            print()
        print('len: ' + str(self.get_len()))

    def get_uris(self):
        uri_set = set()
        for idx in range(self.len):
            graph_query = self.get_graph_query(idx)
            for node in graph_query['nodes']:
                if node['node_type'] != 'literal':
                    uri_set.add(node['id'])
            for edge in graph_query['edges']:
                uri_set.add(edge['relation'])
        return uri_set

    def surface_match_by_annotated_entity_label(self):
        entity_detected = 0
        entity_total = 0
        class_detected = 0
        class_total = 0
        for idx in range(0, self.len):
            question = self.get_question_by_idx(idx)
            graph_query = self.get_graph_query(idx)
            for node in graph_query['nodes']:
                if node['node_type'] != 'entity':
                    continue
                entity_label = node['friendly_name']
                if entity_label.lower() in question.lower() or entity_label.replace('  ', ' ').lower() in question.lower():
                    entity_detected += 1
                entity_total += 1
            for node in graph_query['nodes']:
                if node['node_type'] != 'class':
                    continue
                class_label = node['friendly_name']
                if class_label.lower() in question.lower() or class_label.replace('  ', ' ').lower() in question.lower():
                    class_detected += 1
                class_total += 1
        print('entity linked: ' + str(entity_detected) + ' / ' + str(entity_total) + ' = ' + str(entity_detected / entity_total))
        print('class linked: ' + str(class_detected) + ' / ' + str(class_total) + ' = ' + str(class_detected / class_total))

    def counting_detection_by_keyword(self):
        count_detected = 0
        count_incorrect = 0
        count_total = 0
        count_keyword = ['how many', 'how much', 'the number of', 'what number of', 'total number of', 'the amount of', 'the total numbers of', 'what amount of']
        for idx in range(0, self.len):
            question = self.get_question_by_idx(idx)
            function = self.get_function_by_idx(idx)
            if function == 'count':
                count_total += 1
            detected = False
            for key in count_keyword:
                if function == 'count':
                    if key in question.lower():
                        count_detected += 1
                        detected = True
                        break
                elif key in question.lower():  # not count
                    print('[Incorrect] ' + question)
                    count_incorrect += 1
                    break
            if detected is False and function == 'count':
                print('[Not Detected] ' + question)
        print('count detected: ' + str(count_detected) + ' / ' + str(count_total) + ' = ' + str(count_detected / count_total))
        print('count incorrect: ' + str(count_incorrect))

    def order_detection_by_keyword(self):
        order_detected = 0
        order_incorrect = 0
        order_total = 0
        order_keyword = ['minimum', 'maximum', ' max ', ' min ', 'biggest', 'latest', 'oldest', 'youngest', 'most recent', 'tallest', 'earliest', 'fewest',
                         'smallest', 'longest', 'least', 'largest', 'lightest', 'greatest', 'highest', 'newest', 'fattest']
        for idx in range(0, self.len):
            question = self.get_question_by_idx(idx)
            function = self.get_function_by_idx(idx)
            if function == 'argmax' or function == 'argmin':
                order_total += 1
            detected = False
            for key in order_keyword:
                if function == 'argmax' or function == 'argmin':
                    if key in question.lower():
                        order_detected += 1
                        detected = True
                        break
                elif key in question.lower():  # not order
                    print('[Incorrect] ' + question)
                    order_incorrect += 1
                    break
            if detected is False and (function == 'argmax' or function == 'argmin'):
                print('[Not Detected] ' + question)
        print('order detected: ' + str(order_detected) + ' / ' + str(order_total) + ' = ' + str(order_detected / order_total))
        print('order incorrect: ' + str(order_incorrect))

    def comparison_detection_by_keyword(self):
        comparison_detected = 0
        comparison_incorrect = 0
        comparison_total = 0
        comparison_keyword = ['less than', 'smaller than', 'greater than', 'more than', 'later than', 'at least', 'below', 'at most',
                              'heavier than', 'larger than', 'shorter than', 'below', 'under', 'above']
        comparison_functions = ['<', '<=', '>', '>=']
        for idx in range(0, self.len):
            question = self.get_question_by_idx(idx)
            function = self.get_function_by_idx(idx)
            if function in comparison_functions:
                comparison_total += 1
            detected = False
            for key in comparison_keyword:
                if function in comparison_functions:
                    if key in question.lower() and re.search(r'\d', question):
                        comparison_detected += 1
                        detected = True
                        break
                elif key in question.lower() and re.search(r'\d', question):  # not comparison
                    print('[Incorrect] ' + question)
                    comparison_incorrect += 1
                    break
            if detected is False and function in comparison_functions:
                print('[Not Detected] ' + question)
        print('comparison detected: ' + str(comparison_detected) + ' / ' + str(comparison_total) + ' = ' + str(comparison_detected / comparison_total))
        print('comparison incorrect: ' + str(comparison_incorrect))

    def get_question_with_function(self, function_list: list):
        res = []
        for idx in range(self.len):
            function = self.get_function_by_idx(idx)
            question = self.get_question_by_idx(idx)
            if function in function_list:
                res.append(question)
        return res

    def get_num_entity_class_and_literal(self):
        num_entity_dict = dict()
        num_class_dict = dict()
        num_literal_dict = dict()

        for idx in range(self.len):
            graph_query = self.get_graph_query(idx)
            num_entity = 0
            num_class = 0
            num_literal = 0
            for node in graph_query['nodes']:
                if node['node_type'] == 'entity':
                    num_entity += 1
                elif node['node_type'] == 'class':
                    num_class += 1
                elif node['node_type'] == 'literal':
                    num_literal += 1
            num_entity_dict[num_entity] = num_entity_dict.get(num_entity, 0) + 1
            num_class_dict[num_class] = num_class_dict.get(num_class, 0) + 1
            num_literal_dict[num_literal] = num_literal_dict.get(num_literal, 0) + 1
        return num_entity_dict, num_class_dict, num_literal_dict

    def get_golden_entity_by_idx(self, idx, only_label=False, only_id=False):
        res = []
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'entity':
                res.append(node)
        if only_id and len(res):
            return [node['id'] for node in res]
        if only_label and len(res):
            return [node['friendly_name'] for node in res]
        return res

    def get_num_golden_entity_by_idx(self, idx):
        res = 0
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'entity':
                res += 1
        return res

    def get_golden_class_by_idx(self, idx, only_label=False):
        res = []
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'class':
                res.append(node)
        if only_label and len(res):
            return [node['id'] for node in res]
        return res

    def get_anchor_entity_class_by_idx(self, idx):
        res = set()
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'entity':
                res.add(node['class'])
        return res

    def get_golden_relation_by_idx(self, idx):
        res = set()
        graph_query = self.get_graph_query(idx)
        for edge in graph_query['edges']:
            e = edge['relation']
            res.add(e)
        return list(res)

    def get_golden_relation_domain_by_idx(self, idx) -> set:
        res = set()
        golden_relations = self.get_golden_relation_by_idx(idx)
        for r in golden_relations:
            for domain in fb_domain_dict:
                if r.startswith(domain + '.'):
                    res.add(domain)
        return res

    def get_golden_class_domain_by_idx(self, idx) -> set:
        res = set()
        golden_classes = self.get_golden_class_by_idx(idx)
        for c in golden_classes:
            for domain in fb_domain_dict:
                if c['id'].startswith(domain + '.'):
                    res.add(domain)
        return res

    def get_golden_relation_class_domain_by_idx(self, idx) -> set:
        return self.get_golden_relation_domain_by_idx(idx).union(self.get_golden_class_domain_by_idx(idx))

    def get_all_entities(self):
        res = dict()  # entity friendly_name -> entity
        for idx in range(self.len):
            graph_query = self.get_graph_query(idx)
            for node in graph_query['nodes']:
                if node['node_type'] == 'entity':
                    if node['friendly_name'] not in res:
                        res[node['friendly_name']] = set()
                    res[node['friendly_name']].add(node['id'])
        return res

    def get_all_classes(self):
        res = dict()  # class friendly_name -> class
        for idx in range(self.len):
            graph_query = self.get_graph_query(idx)
            for node in graph_query['nodes']:
                if node['node_type'] == 'class':
                    if node['friendly_name'] not in res:
                        res[node['friendly_name']] = set()
                    res[node['friendly_name']].add(node['class'])
        # reordering
        test_dict_list = sorted(list(res.items()), key=lambda key: len(key[0]), reverse=True)
        res = {ele[0]: ele[1] for ele in test_dict_list}
        return res


def print_question_with_logic_forms(dataloader):
    for idx in range(0, dataloader.len):
        print(dataloader.get_question_by_idx(idx))
        print(dataloader.get_sparql_by_idx(idx))
        print(dataloader.get_s_expression_by_idx(idx))
        print()
    print('len: ' + str(dataloader.get_len()))


def print_question_with_golden_relations(dataloader: GrailQAJsonLoader):
    for idx in range(0, dataloader.len):
        print(dataloader.get_question_by_idx(idx))
        relation = dataloader.get_golden_relation_by_idx(idx)
        print(relation)
        relation_domain = dataloader.get_golden_relation_domain_by_idx(idx)
        print(relation_domain)
        print('#relation: ' + str(len(relation)) + ', #domain: ' + str(len(relation_domain)))
        print()
    print('len: ' + str(dataloader.get_len()))


if __name__ == "__main__":
    train_data = GrailQAJsonLoader(grailqa_train_path)
    dev_data = GrailQAJsonLoader(grailqa_dev_path)

    # print_sparqls(train_data)
    # print_question_with_logic_forms(train_data)
    print_question_with_golden_relations(dev_data)

    # train_data.surface_match_by_annotated_entity_label()
    # dev_data.surface_match_by_annotated_entity_label()
