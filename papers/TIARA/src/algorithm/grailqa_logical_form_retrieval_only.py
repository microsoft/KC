# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.ranking_candidate import LogicalFormRetriever
from utils.config import grailqa_tiara_ranking_dev_path, grailqa_tiara_ranking_test_path, grailqa_dev_path
from utils.file_util import write_json_file
from utils.logic_form_util import execute_s_expr

if __name__ == '__main__':
    ranking_candidate = LogicalFormRetriever(dev_file_path=grailqa_tiara_ranking_dev_path, test_file_path=grailqa_tiara_ranking_test_path)
    res = dict()  # {qid: {'logical_form': logical_form_str, 'answer': mid_list}}
    grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)

    for idx in range(grailqa_dev.len):
        qid = grailqa_dev.get_question_id_by_idx(idx, format='str')
        question = grailqa_dev.get_question_by_idx(idx)
        logical_forms = ranking_candidate.get_logical_form_by_question_id(qid)

        item = dict()
        pred_lf = ''
        pred_ans = []
        if logical_forms is not None and len(logical_forms):
            for logical_form in logical_forms:
                lf, ans = execute_s_expr(logical_form)
                if len(ans) and ans[0] != '0':
                    pred_lf = lf
                    pred_ans = ans
                    break
        item['logical_form'] = pred_lf
        item['answer'] = pred_ans
        res[qid] = item

    # output
    write_json_file('../logs/grailqa_dev_logical_form_retrieval_only.json', res)
