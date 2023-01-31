# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import pickle
import shutil


def pickle_save(obj, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except:
        print('[WARN] pickle save exception: ' + file_path)


def pickle_load(file_path: str, default=None):
    if not os.path.exists(file_path):
        return default
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def read_tsv_as_list(tsv_file_path: str, sep='\t'):
    res = []
    with open(tsv_file_path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            line_split = line.strip('\n').split(sep)
            res.append(line_split)
    return res


def read_list_file(file_path: str):
    res = []
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            res.append(line.strip('\n'))
    return res


def read_json_file(file_path: str, key=None):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='UTF-8') as file:
        try:
            f = json.load(file)
            res = f
        except Exception as e:
            print(e)
            return None
    if key is None or type(res) == dict:
        return res
    else:
        new_res = {}
        for item in res:
            new_res[item[key]] = item
        return new_res


def write_json_file(file_path: str, obj):
    with open(file_path, 'w', encoding='UTF-8') as file:
        json.dump(obj, file)


def read_jsonl_as_list(file_path: str):
    res = []
    with open(file_path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            res.append(json.loads(line.strip('\n')))
    return res


def read_json_as_dict(file_path: str, key: str):
    res = dict()
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = json.load(file)
    for item in data:
        res[item[key]] = item
    return res


def read_jsonl_as_dict(file_path: str, key: str):
    res = dict()
    with open(file_path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            item = json.loads(line.strip('\n'))
            res[item[key]] = item
    return res


def remove_duplicate_lines(file_path):
    lines_seen = set()
    new_file = open(f"{file_path}.out", 'a+', encoding='utf-8')
    file = open(file_path, 'r', encoding='utf-8')
    for line in file:
        if line not in lines_seen:
            new_file.write(line)
            lines_seen.add(line)
    new_file.close()
    file.close()
    shutil.move(f"{file_path}.out", file_path)  # rename the file


def check_dir_exists(dir_path: str):
    if not os.path.exists(dir_path):
        print('[INFO] ' + dir_path + ' does not exits, it has been created')
        os.mkdir(dir_path)
        return False
    return True


def merge_json_list_to_pickle(file_path_list: list, output_path: str):
    res = []
    for file_path in file_path_list:
        data = read_json_file(file_path)
        res += data
    # write_json_file(output_path, res)
    pickle_save(res, output_path)


if __name__ == '__main__':
    merge_json_list_to_pickle(['../dataset/GrailQA/entity_linking_results/grailqa_train_input_top5.jsonl0.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_train_input_top5.jsonl1.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_train_input_top5.jsonl2.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_train_input_top5.jsonl3.json'],
                              '../dataset/GrailQA/entity_linking_results/grailqa_train_input_top5.pkl')
    merge_json_list_to_pickle(['../dataset/GrailQA/entity_linking_results/grailqa_dev_input_top5.jsonl0.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_dev_input_top5.jsonl1.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_dev_input_top5.jsonl2.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_dev_input_top5.jsonl3.json'],
                              '../dataset/GrailQA/entity_linking_results/grailqa_dev_input_top5.pkl')
    merge_json_list_to_pickle(['../dataset/GrailQA/entity_linking_results/grailqa_test_input_top5.jsonl0.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_test_input_top5.jsonl1.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_test_input_top5.jsonl2.json',
                               '../dataset/GrailQA/entity_linking_results/grailqa_test_input_top5.jsonl3.json'],
                              '../dataset/GrailQA/entity_linking_results/grailqa_test_input_top5.pkl')
    print('[INFO] Done')