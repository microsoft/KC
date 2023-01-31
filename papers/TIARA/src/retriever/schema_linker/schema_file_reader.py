# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.config import grailqa_schema_train_path, grailqa_schema_dev_path, grailqa_schema_test_path
from utils.file_util import read_json_file, read_jsonl_as_dict


class SchemaLinker:
    def __init__(self, relation_top_k=50, class_top_k=50,
                 train_file_path=grailqa_schema_train_path, dev_file_path=grailqa_schema_dev_path, test_file_path=grailqa_schema_test_path, file_format='jsonl'):

        self.relations = dict()
        self.classes = dict()
        for file_path in [test_file_path, dev_file_path, train_file_path]:
            if file_path is None:
                continue
            if file_format == None:
                file_format = file_path.split('.')[-1]
            if file_format == 'jsonl':
                res = read_jsonl_as_dict(file_path, key='qid')
            elif file_format == 'json':
                res = read_json_file(file_path, key='qid')

            if res is None:
                print('Schema cannot be read from ', file_path)
                continue
            for qid in res:
                item = res[qid]
                if 'classes' in item and len(item['classes']):
                    c = item['classes'][:class_top_k]
                    if isinstance(c[0], list):
                        c = [item[0] for item in c]
                    self.classes[qid] = c
                if 'relations' in item and len(item['relations']):
                    r = item['relations'][:relation_top_k]
                    if isinstance(r[0], list):
                        r = [item[0] for item in r]
                    self.relations[qid] = r

    def get_relation_by_question_id(self, qid, top_k=10, norm=False):
        res = self.relations.get(qid, None)
        if res is None:
            if type(qid) == int:
                qid = str(qid)
            elif qid.isdigit():
                qid = int(qid)
            res = self.relations.get(qid, None)

        if res is not None:
            res = res[:top_k]
            if norm:
                new_res = []
                for r in res:
                    new_res.append(r.replace('_', ' ').replace('.', ', '))
                res = new_res
        else:
            print('No relation found for qid', qid, type(qid))
        return res

    def get_class_by_question_id(self, qid: int, top_k=10, norm=False):
        res = self.classes.get(qid, None)
        if res is None:
            if type(qid) == int:
                qid = str(qid)
            else:
                qid = int(qid)
            res = self.classes.get(qid, None)

        if res is not None:
            res = res[:top_k]
            if norm:
                new_res = []
                for c in res:
                    new_res.append(c.replace('_', ' ').replace('.', ', '))
                res = new_res
        else:
            print('No class found for qid', qid, type(qid))
        return res


if __name__ == '__main__':
    schema_linker = SchemaLinker()
