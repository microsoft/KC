# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from retriever.entity_linker.webqsp_entity_linker import WebQSPEntityLinker
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import webqsp_train_path, webqsp_pdev_path, webqsp_ptrain_path
from utils.file_util import read_json_file
from utils.metrics import Metrics
from utils.statistics.webqsp_legacy_eval import FindInList


def calculate_entity_p_r_f1(goldAnswerList, predAnswerList):
    if goldAnswerList is None or len(goldAnswerList) == 0:
        if predAnswerList is None or len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0, 0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif predAnswerList is None or len(predAnswerList) == 0:
        return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist = goldAnswerList
        plist = predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry, plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]


def entity_statistics(webqsp_data, entity_linker: WebQSPEntityLinker):
    metric = Metrics()
    for entry in webqsp_data:
        skip = True
        for pidx in range(len(entry['Parses'])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False
        if len(entry['Parses']) == 0 or skip:
            continue
        metric.count()
        qid = entry["QuestionId"]
        pred_ans = entity_linker.get_entities_by_question_id(qid, True)
        if len(entry['Parses']) == 0:
            print('Empty parses in the gold set. Breaking!')
            break

        best_f1 = -9999
        best_recall = -9999
        best_precision = -9999
        for pidx in range(len(entry['Parses'])):
            pidx_ans = set()
            topic_mid = entry['Parses'][pidx]['TopicEntityMid']
            if topic_mid is not None:
                pidx_ans.add(topic_mid)
            constraints = entry['Parses'][pidx]['Constraints']
            if constraints is not None:
                for c in constraints:
                    if c['ArgumentType'] == 'Entity':
                        pidx_ans.add(c['Argument'])

            prec, rec, f1 = calculate_entity_p_r_f1(pidx_ans, pred_ans)
            if f1 > best_f1:
                best_f1 = f1
                best_recall = rec
                best_precision = prec
        metric.add_metric('f1', best_f1)
        metric.add_metric('recall', best_recall)
        metric.add_metric('precision', best_precision)

    print(metric.get_metrics(['precision', 'recall', 'f1']))


if __name__ == '__main__':
    retriever = FreebaseRetriever()
    entity_linker = WebQSPEntityLinker(retriever)

    webqsp_ptrain = read_json_file(webqsp_ptrain_path)
    webqsp_pdev = read_json_file(webqsp_pdev_path)
    webqsp_test = read_json_file(webqsp_train_path)

    entity_statistics(webqsp_pdev, entity_linker)
    entity_statistics(webqsp_test, entity_linker)
