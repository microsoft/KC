# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from collections import OrderedDict
from prettytable import PrettyTable


def eval_webqsp(pred_fn):
    metrics = OrderedDict()
    with open(pred_fn, mode="r", encoding="utf-8") as fp:
        pred_count = 0
        correct_count = 0
        gold_count = 0
        for line in fp:
            sent = json.loads(line.strip())
            topic_entities = set(sent["topic_entities"])
            pred_entities = set([x for x in sent["qids"] if x != "NIL"])
            pred_count += len(pred_entities)
            gold_count += len(topic_entities)
            correct_count += len(pred_entities & topic_entities)

        p = correct_count / pred_count
        r = correct_count / gold_count
        f1 = 2.0 / (1.0 / p + 1.0 / r)

        metrics["all"] = {"gold_instances": gold_count,
                          "pred_instances": pred_count,
                          "correct_instances": correct_count,
                          "precision": p * 100,
                          "recall": r * 100,
                          "f1": f1 * 100}
        return metrics


def eval_webqsp_prior(pred_fn):
    metrics = OrderedDict()
    with open(pred_fn, mode="r", encoding="utf-8") as fp:
        pred_count = 0
        correct_count = 0
        gold_count = 0
        for line in fp:
            sent = json.loads(line.strip())
            topic_entities = set(sent["topic_entities"])
            pred_entities = set()
            for ent in sent["prior_baseline"]:
                if ent[0] == "entity" and ent[3] != "NIL":
                    pred_entities.add(ent[3])
            pred_count += len(pred_entities)
            gold_count += len(topic_entities)
            correct_count += len(pred_entities & topic_entities)

        p = correct_count / pred_count
        r = correct_count / gold_count
        f1 = 2.0 / (1.0 / p + 1.0 / r)

        metrics["all"] = {"gold_instances": gold_count,
                          "pred_instances": pred_count,
                          "correct_instances": correct_count,
                          "precision": p * 100,
                          "recall": r * 100,
                          "f1": f1 * 100}
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_fn",
                        type=str,
                        required=True)
    parser.add_argument("--model", type=str, default="bootleg")
    args = parser.parse_args()

    if args.model == "bootleg":
        metrics = eval_webqsp(args.pred_fn)
    else:
        metrics = eval_webqsp_prior(args.pred_fn)

    results = PrettyTable()
    results.field_names = [
        "Partition",
        "Precision",
        "Recall",
        "F1",
        "#GoldSupport",
        "#PredSupport",
        "#Correct"
    ]

    for part in metrics:
        results.add_row([part,
                         '{:.2f}'.format(metrics[part]['precision']),
                         '{:.2f}'.format(metrics[part]['recall']),
                         '{:.2f}'.format(metrics[part]['f1']),
                         metrics[part]['gold_instances'],
                         metrics[part]['pred_instances'],
                         metrics[part]['correct_instances']])

    print(results)
