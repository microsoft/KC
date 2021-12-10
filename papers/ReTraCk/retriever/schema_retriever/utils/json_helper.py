# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os


def from_str(s):
    return json.loads(s)


def from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return from_str(''.join(f.readlines()))
    except Exception:
        print("Absolute path: " + os.path.abspath(path))
        raise


def to_str(d, indent=None):
    return json.dumps(d, indent=indent)


def to_file(d, path, indent=None):

    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(path, "w", encoding="utf-8") as f:
        f.write(to_str(d, indent))


def from_jsonl(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            res.append(json.loads(line))
    return res
