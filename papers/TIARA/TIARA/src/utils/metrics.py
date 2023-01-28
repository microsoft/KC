# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random


class Metrics:
    num_sample: int
    data: dict

    def __init__(self):
        self.num_sample = 0
        self.data = dict()

    def add_metric(self, key, value):
        if key not in self.data:
            self.data[key] = 0
        self.data[key] += value

    def count(self):
        self.num_sample += 1

    def get_metric(self, key, r=4):
        if key not in self.data:
            self.data[key] = 0
            return self.data[key]
        if r is not None:
            return round(self.data[key] / self.num_sample, r)
        return self.data[key] / self.num_sample, r

    def get_metrics(self, keys: list):
        res = ''
        for key in keys:
            value = self.get_metric(key)
            res += key + ': ' + str(value) + ', '
        return res[:-2]


def get_precision(golden, pred):
    if pred is None or len(pred) == 0:
        return 0.0
    golden_set = to_set(golden)
    pred_set = to_set(pred)
    return len(golden_set.intersection(pred_set)) / len(pred_set)


def get_recall(golden, pred):
    if golden is None or len(golden) == 0:
        return 0.0
    golden_set = to_set(golden)
    pred_set = to_set(pred)
    return len(golden_set.intersection(pred_set)) / len(golden_set)


def get_precision_at_1_by_random(golden, pred):
    if type(golden) == bool or type(golden) == int:  # boolean or count
        if golden == pred:
            return 1
        else:
            return 0
    if pred is None or len(pred) == 0 or golden is None or len(golden) == 0:  # prediction or golden is empty
        return 0
    return int((random.sample(pred, 1)[0]) in golden)


def get_f1_score_by_pr(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def get_hits(golden, pred, k=1):
    total = 0
    count = 0
    for i in range(0, min(k, len(pred))):
        total += 1
        if pred[i] in golden:
            count += 1
    if total == 0:
        return 0
    return count / total


def get_em_acc(golden, pred):
    if golden == pred:
        return 1.0
    return 0.0


def get_f1_score(golden, pred):
    p = get_precision(golden, pred)
    r = get_recall(golden, pred)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def get_p_r_f1_score_webqsp(golden, pred):
    if type(golden) == bool or type(golden) == int:
        if golden == pred:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    if golden is None or len(golden) == 0:
        if pred is None or len(pred) == 0:  # both empty
            return 1.0, 1.0, 1.0
        else:  # golden is empty, prediction is not
            return 0.0, 1.0, 0.0
    elif pred is None or len(pred) == 0:  # golden is not empty, prediction is empty
        return 1.0, 0.0, 0.0

    p = get_precision(golden, pred)
    r = get_recall(golden, pred)
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * p * r / (p + r)
    return p, r, f1


def get_p_r_f1_score(golden, pred):
    if type(golden) == bool or type(golden) == int:
        if golden == pred:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    p = get_precision(golden, pred)
    r = get_recall(golden, pred)
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * p * r / (p + r)
    return p, r, f1


def to_set(s):
    if s is None:
        return set()
    if isinstance(s, set):
        return s
    return set(s)
