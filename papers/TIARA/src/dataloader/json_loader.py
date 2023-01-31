# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class JsonLoader:
    def get_dataset_split(self):
        if 'train' in self.file_path:
            return 'train'
        elif 'dev' in self.file_path or 'val' in self.file_path:
            return 'dev'
        elif 'test' in self.file_path:
            return 'test'
        return ''

    def get_golden_relations(self):
        relation_set = set()
        assert self.len is not None and self.len != 0
        for idx in range(0, self.len):
            golden_relation = self.get_golden_relation_by_idx(idx)
            if golden_relation is None or len(golden_relation) == 0:
                continue
            for r in golden_relation:
                relation_set.add(r)
        return relation_set
