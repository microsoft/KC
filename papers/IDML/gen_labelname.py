# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, json

def generate_labels(input_fn, output_fn):
    all_labels = []
    with open(input_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            label = json.loads(line)["label"]
            if label not in all_labels:
                all_labels.append(label)
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines([x + "\n" for x in all_labels])
    return

if __name__ == "__main__":
    generate_labels("./data/BANKING77/full.jsonl", "./data/BANKING77/labels.txt")
    generate_labels("./data/HWU64/full.jsonl", "./data/HWU64/labels.txt")
    generate_labels("./data/Liu/full.jsonl", "./data/Liu/labels.txt")
    generate_labels("./data/OOS/full.jsonl", "./data/OOS/labels.txt")
