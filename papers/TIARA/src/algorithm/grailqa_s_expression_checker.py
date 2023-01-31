# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import T5Tokenizer

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from utils.config import grailqa_dev_path, grailqa_train_path
from utils.trie_util.schema_tokenization import build_token_trie


class GrailQASExpressionChecker:

    def __init__(self):
        self.class_trie, self.relation_trie = build_token_trie()
        self.class_start = self.class_trie.query_child([])
        self.relation_start = self.class_trie.query_child([])

        self.last_schema = []

        self.operator_tokens = ['▁(', 'X', 'g', 'CO', 'AND', 'IN', 't', 'le', ')', 'G', 'AR', 'MIN', 'A', 'GM', 'JO', 'R', 'l', 'ge', 'UNT']
        self.literal_tokens = ['^^', 'http', '://', 'www', '.', 'w', '3.', 'org', '/', '2001', '/', 'X', 'MLS', 'chem', 'a', '#']

    def valid_tokens(self, input_tokens: list, verbose=False):
        self.last_schema = []
        input_len = len(input_tokens)
        if input_tokens[-1] == '<pad>':  # the first token is <pad>
            return ['▁(']
        elif input_len == 2:
            return ['CO', 'AR', 'AND']
        elif input_tokens[-1] == 'AND':
            return self.operator_tokens + self.class_trie.query_child([])
        elif (input_len >= 3 and input_tokens[-3:] == ['AR', 'G', 'MIN']) or (input_len >= 4 and input_tokens[-4:] == ['AR', 'GM', 'A', 'X']):
            return ['▁('] + self.class_trie.query_child([])
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'JO']:
            return ['IN']
        elif input_len >= 2 and input_tokens[-2:] == ['JO', 'IN']:
            return self.relation_trie.query_child([]) + ['▁(']
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'R']:
            return self.relation_trie.query_child([])
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'CO']:
            return ['UNT']
        elif input_len >= 2 and input_tokens[-2:] == ['▁(', 'AR']:
            return ['G', 'GM']
        elif input_len >= 3 and input_tokens[-3:] == ['▁(', 'CO', 'UNT']:
            return ['▁(']
        elif input_len >= 4 and input_tokens[-4:] == ['▁(', 'CO', 'UNT', '▁(']:
            return ['AND']
        elif (input_len >= 2 and input_tokens[-2:] == ['▁(', 'le']) or (input_len >= 3 and input_tokens[-3:] == ['▁(', 'l', 't']) or (
                input_len >= 2 and input_tokens[-2:] == ['▁(', 'ge']) or (input_len >= 3 and input_tokens[-3:] == ['▁(', 'g', 't']):
            return self.relation_trie.query_child([])
        elif input_tokens[-1] == '▁(':
            return ['CO', 'AR', 'AND', 'R', 'le', 'ge', 'l', 'g', 'JO']
        elif input_tokens[-1] in self.literal_tokens:
            if input_tokens[-1] != '#' and '^^' in input_tokens:
                start_index = input_tokens.index('^^')
                cur_literal_token_len = len(input_tokens) - start_index
                if input_tokens[start_index:] == self.literal_tokens[:cur_literal_token_len]:
                    return [self.literal_tokens[cur_literal_token_len]]

        if input_tokens[-1] == '▁':  # start of a entity, a schema, or a literal
            if input_tokens[-3:] == ['JO', 'IN', '▁']:
                return self.relation_trie.query_child(['▁'])
            elif input_tokens[-2:] == ['AND', '▁']:
                return self.class_trie.query_child(['▁'])
            return None
        last_schema_type = None
        last_schema_index = None
        i = len(input_tokens) - 1
        for i in range(len(input_tokens) - 1, -1, -1):  # inverse traversal to find the last schema
            if input_tokens[i] in ['AND', 'X', 'MIN']:
                last_schema_type = 'class'
                break
            elif input_tokens[i] in ['R', 'IN']:
                last_schema_type = 'relation'
                break
            if input_tokens[i] in self.class_start or input_tokens[i] in self.relation_start:
                last_schema_index = i
            if '▁' in input_tokens[i]:
                break
        if last_schema_type is None and last_schema_index is not None:  # get the schema type
            for j in range(i - 1, -1, -1):
                if input_tokens[j] in ['AND', 'X', 'MIN']:
                    mid_tokens = input_tokens[j + 1:i]
                    if len(set(mid_tokens).difference(self.operator_tokens)) == 0:  # all middle tokens are operators
                        if 'l' in mid_tokens or 'le' in mid_tokens or 'g' in mid_tokens or 'ge' in mid_tokens:
                            last_schema_type = 'relation'
                        else:
                            last_schema_type = 'class'
                        break
                    else:  # there are some non-operator tokens
                        last_schema_type = 'relation'
                        break
                elif input_tokens[j] in ['R', 'IN']:
                    last_schema_type = 'relation'
                    break

        if last_schema_index is not None:  # note that two schemas may appear consecutively
            trie = self.class_trie if last_schema_type == 'class' else self.relation_trie
            self.last_schema = input_tokens[last_schema_index:]
            if self.last_schema[:2] == ['▁', 'm'] or self.last_schema[:2] == ['▁', 'g'] or any(ch.isdigit() for ch in self.last_schema[-1]):  # entity or literal
                return None
            valid_schema_tokens, node_count = trie.query_child(self.last_schema, True)
            if len(valid_schema_tokens) > 0:  # schema may not be finished
                if node_count:
                    # 1. operator, 2. continue schema, 3. (relation) -> entity / literal, 4. (class) -> relation
                    if verbose:
                        print('[INFO] schema may or may not be finished')
                    return None
                else:  # schema indeed not finished
                    return valid_schema_tokens
            else:
                # if input_tokens[1:3] == ['▁(', 'AR'] and 'JO' not in input_tokens:  # operator, entity, relation
                #     return self.operator_tokens + ['▁'] + self.relation_trie.query_child([])
                if verbose:  # literal also possible
                    print('[INFO] valid schema token is empty, current schema is finished')
                return None
        if verbose:
            print('[INFO] no rule for now')
        return None


def checker_test_by_golden(dataloader, verbose=False):
    for idx in range(0, dataloader.len):
        golden_s_expression = dataloader.get_s_expression_by_idx(idx)
        golden_tokens = ['<pad>'] + tokenizer.tokenize(golden_s_expression)

        last_valid_tokens = None

        for i in range(0, len(golden_tokens)):
            t = golden_tokens[:i + 1]
            valid_tokens = schema_checker.valid_tokens(t, verbose=verbose)

            if last_valid_tokens is not None and len(t) and t[-1] not in last_valid_tokens:
                print('[ERROR] checking failed'.format(t))
            last_valid_tokens = valid_tokens
            if verbose:
                print('[' + str(idx) + ']', t)
                print('last schema:', schema_checker.last_schema)
                print('valid:', valid_tokens)
                print()


if __name__ == '__main__':
    grail_train_data = GrailQAJsonLoader(grailqa_train_path)
    grail_dev_data = GrailQAJsonLoader(grailqa_dev_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    schema_checker = GrailQASExpressionChecker()

    checker_test_by_golden(grail_train_data)
    checker_test_by_golden(grail_dev_data)
    print('checker test finished')
