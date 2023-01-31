# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os

sys.path.append('.')
sys.path.append('..')

import json
from utils.file_util import write_json_file


def main():
    if len(sys.argv) != 3:
        print("Usage: python webqsp_evaluate.py goldData predAnswers")
        sys.exit(-1)

    pred_answers = open(sys.argv[2]).readlines()  # jsonl

    legacy_format = []
    for item in pred_answers:
        item = json.loads(item)
        legacy_format.append({"QuestionId": item["qid"], "Answers": item["answer"]})

    tmp_filename = "../logs/tmp_legacy_pred.json"
    write_json_file(tmp_filename, legacy_format)

    legacy_eval_path = 'utils/statistics/webqsp_legacy_eval.py'
    os.system("python2 {} {} {}".format(legacy_eval_path, sys.argv[1], tmp_filename))
    os.remove(tmp_filename)


if __name__ == "__main__":
    main()
