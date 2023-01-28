# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import T5Tokenizer

from dataloader.webqsp_json_loader import WebQSPJsonLoader
from utils.config import webqsp_train_path, webqsp_test_path, webqsp_question_2hop_relation_path, webqsp_relation_trie_path
from utils.file_util import read_list_file, pickle_load, pickle_save
from utils.trie_util.trie import Trie


def build_webqsp_relation_trie():
    relation_trie = pickle_load(webqsp_relation_trie_path)
    if relation_trie is None:
        relations = read_list_file('../dataset/webqsp_searched_relations.txt')
        if relations is None:
            question_relations_map = pickle_load(webqsp_question_2hop_relation_path)
            assert question_relations_map is not None
            relations = set()
            for question_id, relations_list in question_relations_map.items():
                relations.update(relations_list)
            golden_relations = webqsp_train_data.get_golden_relations()
            relations.update(golden_relations)
            pickle_save(relations, '../dataset/webqsp_searched_relations.txt')

        relation_trie = Trie()
        for r in relations:
            relation_trie.insert(tokenizer.tokenize(r))
        pickle_save(relation_trie, webqsp_relation_trie_path)
    return relation_trie


class WebQSPSExpressionChecker:
    def __init__(self):
        self.relation_trie = build_webqsp_relation_trie()
        self.relation_start = self.relation_trie.query_child([])

        self.last_schema = []
        self.operator_tokens = ['▁(', 'X', 'AND', 'IN', ')', 'G', 'AR', 'MIN', 'A', 'GM', 'JO', 'R']
        self.literal_tokens = ['^^', 'http', '://', 'www', '.', 'w', '3.', 'org', '/', '2001', '/', 'X', 'MLS', 'chem', 'a', '#']
        self.special_tokens = ['</s>', '<pad>', '<unk>']

    def valid_tokens(self, input_tokens: list, verbose=False):
        self.last_schema = []
        input_len = len(input_tokens)
        if input_tokens[-1] == '<pad>':  # the first token is <pad>
            return ['▁(']
        elif input_len == 2:
            return ['AR', 'AND', 'JO']
        elif input_tokens[-1] == ')':  # entity (mid/gid), relation (ARGMIN/ARGMAX), '▁(', ')', special tokens
            valid = ['▁(', ')', '▁'] + self.special_tokens
            if 'AR' in input_tokens:
                return valid + self.relation_start
            return valid
        elif input_tokens[-1] == 'AND':
            return ['▁(']
        elif (input_len >= 3 and input_tokens[-3:] == ['AR', 'G', 'MIN']) or (input_len >= 4 and input_tokens[-4:] == ['AR', 'GM', 'A', 'X']):
            return ['▁(']
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'JO']:
            return ['IN']
        elif input_len >= 2 and input_tokens[-2:] == [')', '▁']:  # entity, relation (ARGMIN/MAX), ')', special tokens
            valid = ['m', 'g', ')'] + self.special_tokens
            if 'AR' in input_tokens:
                return valid + self.relation_trie.query_child(['▁'])
            return valid
        elif input_len >= 2 and input_tokens[-2:] == ['JO', 'IN']:
            return self.relation_start + ['▁(']
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'R']:
            return self.relation_start
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'AR']:
            return ['G', 'GM']
        elif input_tokens[-1] == '▁(':
            return ['AR', 'AND', 'R', 'JO']
        elif input_tokens[-1] in self.literal_tokens:
            if input_tokens[-1] != '#' and '^^' in input_tokens:
                start_index = input_tokens.index('^^')
                cur_literal_token_len = len(input_tokens) - start_index
                if input_tokens[start_index:] == self.literal_tokens[:cur_literal_token_len]:
                    return [self.literal_tokens[cur_literal_token_len]]

        if input_tokens[-1] == '▁':  # start of a entity, a relation, or a literal
            if input_tokens[-3:] == ['JO', 'IN', '▁'] or input_tokens[-3:] == ['(', 'R', '▁']:
                return self.relation_trie.query_child(['▁'])
            return None

        last_schema_index = None
        for i in range(len(input_tokens) - 1, -1, -1):  # inverse traversal to find the last schema
            if input_tokens[i] in self.relation_start:
                last_schema_index = i
            if '▁' in input_tokens[i]:
                break
            if input_tokens[i] in [')', 'IN', 'R']:
                last_schema_index = i + 1
                break

        if last_schema_index is not None and last_schema_index < len(input_tokens):
            self.last_schema = input_tokens[last_schema_index:]
            if self.last_schema[:2] == ['▁', 'm'] or self.last_schema[:2] == ['▁', 'g'] or any(ch.isdigit() for ch in self.last_schema[-1]):  # entity or literal
                return None
            valid_tokens, node_count = self.relation_trie.query_child(self.last_schema, True)
            if len(valid_tokens) > 0:
                if node_count:
                    return None
                else:  # schema indeed not finished
                    if ('from' in valid_tokens and 'to' in valid_tokens) or ('start' in valid_tokens and 'end' in valid_tokens):
                        valid_tokens.append('time')
                    elif input_tokens[-1] == 'time':
                        valid_tokens.append('_')
                    elif input_tokens[-2:] == ['time', '_']:
                        valid_tokens.append('m')
                    elif input_tokens[-3:] == ['time', '_', 'm']:
                        valid_tokens.append('a')
                    elif input_tokens[-4:] == ['time', '_', 'm', 'a']:
                        valid_tokens.append('cro')
                    return valid_tokens
            else:
                return None
        return None


def checker_test_by_golden(dataloader, verbose=False):
    for idx in range(0, dataloader.len):
        golden_s_expression = dataloader.get_s_expression_by_idx(idx)
        if golden_s_expression is None or golden_s_expression == 'null':
            continue
        golden_tokens = ['<pad>'] + tokenizer.tokenize(golden_s_expression)

        last_valid_tokens = None

        for i in range(0, len(golden_tokens)):
            t = golden_tokens[:i + 1]
            valid_tokens = schema_checker.valid_tokens(t, verbose=verbose)

            if last_valid_tokens is not None and len(t) and t[-1] not in last_valid_tokens:
                print('[ERROR] checking failed: {}'.format(t))
            last_valid_tokens = valid_tokens
            if verbose:
                print('[' + str(idx) + ']', t)
                print('last schema:', schema_checker.last_schema)
                print('valid:', valid_tokens)
                print()


if __name__ == '__main__':
    webqsp_train_data = WebQSPJsonLoader(webqsp_train_path)
    webqsp_test_data = WebQSPJsonLoader(webqsp_test_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    schema_checker = WebQSPSExpressionChecker()

    # checker_test_by_golden(webqsp_train_data, verbose=False)
    checker_test_by_golden(webqsp_test_data, verbose=False)
    print('checker test finished')
