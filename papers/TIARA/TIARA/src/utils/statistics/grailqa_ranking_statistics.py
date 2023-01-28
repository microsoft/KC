# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from tqdm import tqdm

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.ranking_candidate import LogicalFormRetriever
from utils.config import grailqa_rng_ranking_dev_path, grailqa_rng_ranking_train_path, grailqa_dev_path, grailqa_train_path, grailqa_rng_ranking_test_path
from utils.logic_form_util import execute_s_expr


def lf_statistics(dataloader: GrailQAJsonLoader):
    predictions = dict()
    total = dict()
    ranks = dict()
    for idx in tqdm(range(dataloader.len)):
        qid = dataloader.get_question_id_by_idx(idx)
        level = dataloader.get_level_by_idx(idx)
        logical_forms = ranking_candidate.get_logical_form_by_question_id(qid)
        pred_lf = ''
        pred_ans = []
        rank = 0
        if logical_forms is not None:
            for lf in logical_forms:
                rank += 1
                lf, ans = execute_s_expr(lf)
                if len(ans) == 0 or ans == ['0']:
                    continue
                pred_lf = lf
                pred_ans = ans
                break
            ranks[level] = ranks.get(level, 0) + rank
            total[level] = total.get(level, 0) + 1
        predictions[qid] = {'logical_form': pred_lf, 'answer': pred_ans}

    with open('../logs/grailqa_ranking_lf_predictions.json', 'w') as f:
        json.dump(predictions, f)
    for key in ranks:
        print(key, ranks[key] / total[key])
        print(key, total[key])


if __name__ == '__main__':
    params = dict()
    params['lf_train'] = grailqa_rng_ranking_train_path
    params['lf_dev'] = grailqa_rng_ranking_dev_path
    params['lf_test'] = grailqa_rng_ranking_test_path

    ranking_candidate = LogicalFormRetriever(train_file_path=params['lf_train'], dev_file_path=params['lf_dev'], test_file_path=params['lf_test'])

    grailqa_train = GrailQAJsonLoader(grailqa_train_path)
    grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)

    lf_statistics(grailqa_dev)
