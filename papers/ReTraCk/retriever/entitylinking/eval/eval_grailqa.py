# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from collections import OrderedDict
from prettytable import PrettyTable


def eval_grailqa(gold_fn, pred_fn):
    metrics = OrderedDict()

    with open(gold_fn, encoding='utf-8', mode='r') as fp:
        level_dict = dict()
        gold = {"i.i.d.": set(), "compositional": set(),
                "zero-shot": set(), "all": set()}
        for line in fp:
            sent = json.loads(line.strip())
            level_dict[sent["qid"]] = sent["level"]
            for node in sent["graph_query"]["nodes"]:
                if node["node_type"] == "entity":
                    gold[sent["level"]].add((sent["qid"], node["offset"][0], node["offset"][1], node["id"]))
                    gold["all"].add((sent["qid"], node["offset"][0], node["offset"][1], node["id"]))

    with open(pred_fn, encoding='utf-8', mode='r') as fp:
        pred = {"i.i.d.": set(), "compositional": set(),
                "zero-shot": set(), "all": set()}
        for line in fp:
            sent = json.loads(line.strip())
            for i, pred_qid in enumerate(sent["qids"]):
                if pred_qid != "-1" and pred_qid != "NIL":
                    pred[level_dict[sent["qid"]]].add((sent["qid"], sent["spans"][i][0], sent["spans"][i][1], pred_qid))
                    pred["all"].add((sent["qid"], sent["spans"][i][0], sent["spans"][i][1], pred_qid))

        for level in gold:
            P, R = len(pred[level] & gold[level]) / len(pred[level]), len(pred[level] & gold[level]) / len(gold[level])
            F1 = 2 / (1.0 / P + 1.0 / R)
            metrics[level] = {"gold_instances": len(gold[level]),
                              "pred_instances": len(pred[level]),
                              "correct_instances": len(pred[level] & gold[level]),
                              "precision": P * 100,
                              "recall": R * 100,
                              "f1": F1 * 100}
    return metrics


def eval_grailqa_prior(gold_fn, pred_fn):
    metrics = OrderedDict()

    with open(gold_fn, encoding='utf-8', mode='r') as fp:
        level_dict = dict()
        gold = {"i.i.d.": set(), "compositional": set(),
                "zero-shot": set(), "all": set()}
        for line in fp:
            sent = json.loads(line.strip())
            level_dict[sent["qid"]] = sent["level"]
            for node in sent["graph_query"]["nodes"]:
                if node["node_type"] == "entity":
                    gold[sent["level"]].add((sent["qid"], node["offset"][0], node["offset"][1], node["id"]))
                    gold["all"].add((sent["qid"], node["offset"][0], node["offset"][1], node["id"]))

    with open(pred_fn, encoding='utf-8', mode='r') as fp:
        pred = {"i.i.d.": set(), "compositional": set(),
                "zero-shot": set(), "all": set()}
        for line in fp:
            sent = json.loads(line.strip())
            for ent in sent["prior_baseline"]:
                if ent[0] == "entity" and ent[3] != "NIL":
                    pred[level_dict[sent["qid"]]].add((sent["qid"],  ent[1], ent[2], ent[3]))
                    pred["all"].add((sent["qid"], ent[1], ent[2], ent[3]))

        for level in gold:
            P, R = len(pred[level] & gold[level]) / len(pred[level]), len(pred[level] & gold[level]) / len(gold[level])
            F1 = 2 / (1.0 / P + 1.0 / R)
            metrics[level] = {"gold_instances": len(gold[level]),
                              "pred_instances": len(pred[level]),
                              "correct_instances": len(pred[level] & gold[level]),
                              "precision": P * 100,
                              "recall": R * 100,
                              "f1": F1 * 100}
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_fn",
                        type=str,
                        required=True)
    parser.add_argument("--pred_fn",
                        type=str,
                        required=True)
    parser.add_argument("--model", type=str, default="bootleg")
    args = parser.parse_args()
    if args.model == "prior":
        metrics = eval_grailqa_prior(args.gold_fn, args.pred_fn)
    else:
        metrics = eval_grailqa(args.gold_fn, args.pred_fn)


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
