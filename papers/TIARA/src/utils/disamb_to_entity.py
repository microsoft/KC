# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
import argparse

from utils.file_util import read_json_file, write_json_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev', help='dev or test', choices=['dev', 'test'])
    args = parser.parse_args()

    if args.split == 'dev':
        candidate_entities = read_json_file('retriever/outputs/grail_dev_entities.json')
        disamb_index = read_json_file('retriever/results/disamb/grail_dev/predictions.json')
    elif args.split == 'test':
        candidate_entities = read_json_file('retriever/outputs/grail_test_entities.json')
        disamb_index = read_json_file('retriever/results/disamb/grail_test/predictions.json')

    res = {}
    for qid in candidate_entities:
        res[qid] = {'entities': {}}
        for mention_idx in range(0, len(candidate_entities[qid])):
            disamb_key = qid + '-' + str(mention_idx)
            if disamb_key not in disamb_index:
                print('Missing disamb key: ' + disamb_key + ', #entity: ' + str(len(candidate_entities[qid][mention_idx])))
                continue
            entity = candidate_entities[qid][mention_idx][disamb_index[disamb_key]]
            res[qid]['entities'][entity['id']] = {'mention': entity['mention'], 'label': entity['label']}

    write_json_file('retriever/outputs/tiara_' + args.split + '_el_results.json', res)
    print('file saved to retriever/outputs/tiara_' + args.split + '_el_results.json')
