import os
import csv
import json
import argparse
from collections import defaultdict

# convert CB data format
def convert_cb(input_dir, output_dir):
    for input_name, output_name in zip(["train.jsonl", "val.jsonl"], ["train.tsv", "dev.tsv"]):
        data = ["\t".join(["idx", "premise", "hypothesis", "label"]) + "\n"]
        label_cnt_mp = defaultdict(int)
        with open(os.path.join(input_dir, input_name), mode="r", encoding="utf-8") as fp:
            for line in fp:
                tmp = json.loads(line)
                label = "entailment" if tmp["label"] == "entailment" else "not_entailment"
                label_cnt_mp[label] += 1
                data.append("\t".join([str(tmp["idx"]), tmp["premise"], tmp["hypothesis"], label]) + "\n")
        with open(os.path.join(output_dir, output_name), mode="w", encoding="utf-8") as fp:
            fp.writelines(data)
        print(label_cnt_mp)
    return

# convert MNLI data format
def convert_mnli(input_dir, output_dir):
    for input_name, output_name in zip(["train.tsv", "dev_matched.tsv", "dev_mismatched.tsv"],
                ["train.tsv", "dev.tsv", "dev_mismatched.tsv"]):
        label_cnt_mp = defaultdict(int)
        data = ["\t".join(["idx", "premise", "hypothesis", "label"]) + "\n"]
        with open(os.path.join(input_dir, input_name), mode="r", encoding="utf-8") as fp:
            first = True
            for line in fp:
                if first:
                    first = False
                    continue
                tmp = line.strip("\n").split("\t")
                if not(len(tmp) == 12 or len(tmp) == 16):
                    print(tmp)
                    print(len(tmp))
                assert len(tmp) == 12 or len(tmp) == 16
                label = "entailment" if tmp[-1] == "entailment" else "not_entailment"
                label_cnt_mp[label] += 1
                data.append("\t".join([tmp[0], tmp[8], tmp[9], label]) + "\n")
        with open(os.path.join(output_dir, output_name), mode="w", encoding="utf-8") as fp:
            fp.writelines(data)
        print(label_cnt_mp)
    return

# get merge debias data format 
def get_augmented_mnli(mnli_fn, invs_fn, output_fn):
    data = []
    label_cnt_mp = defaultdict(int)
    with open(mnli_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            data.append(line)   
            label_cnt_mp[line.strip("\n").split("\t")[-1]] += 1
    with open(invs_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            tmp = line.strip("\n").split("\t")
            assert len(tmp) == 12
            label = "entailment" if tmp[-1] == "entailment" else "not_entailment"
            data.append("\t".join([tmp[0], tmp[8], tmp[9], label]) + "\n")
            label_cnt_mp[label] += 1
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(data)
    print(label_cnt_mp)
    return

def convert_sst(sst_fn, output_fn):
    data = []
    first = True
    with open(sst_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            if first:
                first = False
                continue
            tmp = line.strip("\n").split("\t")
            assert len(tmp) == 2
            if tmp[1] == "0":
                label = "negative"
            else:
                label = "positive"
            data.append("{}\t{}\n".format(tmp[0], label))
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(data)
    return            

def convert_agnews(input_fn, output_fn):
    data = []
    labels = ["politics", "sports", "business", "technology"]
    with open(input_fn, mode="r", encoding="utf-8") as fp:
        csv_reader = csv.reader(fp, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for line in csv_reader:
            assert len(line) == 3
            label_id = int(line[0]) - 1
            sent = line[1] + ". " + line[2]
            data.append("{}\t{}\n".format(sent, labels[label_id]))
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(data)
    return       

def convert_snips(input_fn, output_fn):
    data = []
    with open(input_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            tmp = line.strip("\n").split("\t")
            sent = " ".join(tmp[:-1])
            sent = " ".join(sent.split())
            label = tmp[-1]
            data.append("{}\t{}\n".format(sent, label))
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(data)
    return               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mnli", type=str)
    parser.add_argument("--input_fn", default=None, type=str)
    parser.add_argument("--input_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--mnli_fn", default=None, type=str)
    parser.add_argument("--aug_fn", default=None, type=str)
    parser.add_argument("--aug_mnli_fn", default=None, type=str)
    args = parser.parse_args()
    if args.task == "mnli":
        convert_mnli(args.input_dir, args.output_dir)
    elif args.task == "cb":
        convert_cb(args.input_dir, args.output_dir)
    elif args.task == "mnli-da":
        get_augmented_mnli(args.mnli_fn, args.aug_fn, args.aug_mnli_fn)
    elif args.task == "sst":
        convert_sst(args.input_fn, os.path.join(args.output_dir, "sst.txt"))
    elif args.task == "agnews":
        convert_agnews(args.input_fn, os.path.join(args.output_dir, "agnews.txt"))
    elif args.task == "snips":
        convert_snips(args.input_fn, os.path.join(args.output_dir, "snips.txt"))        
    else:
        raise ValueError("task should be one of [mnil, cb, mnli-data, mnli-cb]")
    
