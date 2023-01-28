# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')

import argparse
import json

from utils.file_util import read_json_file


def merge_class_relation_to_jsonline(class_file_path, relation_file_path, output_file_path):
    """
    Merge class and relation file into one file.
    """
    class_data = read_json_file(class_file_path, key='qid')
    relation_data = read_json_file(relation_file_path, key='qid')

    assert class_data is not None
    assert relation_data is not None
    assert len(class_data) == len(relation_data)

    with open(output_file_path, 'w') as f:
        for key in class_data:
            item = {'qid': key, 'question': class_data[key]['question'],
                    'classes': class_data[key]['classes'], 'relations': relation_data[key]['relations']}
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_path', type=str, default='../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev_class.json')
    parser.add_argument('--relation_path', type=str, default='../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev_relation.json')
    parser.add_argument('--output_path', type=str, default='../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev.jsonl')
    args = parser.parse_args()

    merge_class_relation_to_jsonline(args.class_path, args.relation_path, args.output_path)
    print('Merging done')
