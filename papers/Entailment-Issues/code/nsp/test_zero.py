# coding=utf-8
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function


import argparse
import csv
import logging
import os
import random
import sys
import codecs
import json
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from util import *
from metric import *
# import clf_distill_loss_functions

from scipy.special import softmax

from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers.modeling_bert import BertForSequenceClassification, BertConfig, BertForNextSentencePrediction
# from bert_distill import *
from transformers.tokenization_bert import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def cal_accuracy(entail_probs, examples, hypo_type_list, gold_label_list):
    assert entail_probs.shape[0] == len(examples)
    text_num = int(len(examples) / len(hypo_type_list))
    pid = 0
    hit_size = 0
    pred_label_list = []
    for i in range(text_num):
        max_prob = -100.0
        max_type = None
        for hypo, type in hypo_type_list:
            if entail_probs[pid] > max_prob:
                max_prob = entail_probs[pid]
                max_type = type
            pid += 1
        pred_label_list.append(max_type)
        if max_type == gold_label_list[i]:
            hit_size += 1
    return pred_label_list, hit_size / text_num

def eval(model, dataloader, device, num_labels, return_logits=False):
    model.eval()
    logger.info("***** Running evaluation *****")
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    print('Evaluating...')
    for idx, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": segment_ids,
            }
            logits = model(**inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
    preds = preds[0]
    #entailment prob
#     print(preds.shape)
#     print(preds[0])
    pred_probs = softmax(preds, axis=1)[:, 0]
    if return_logits:
        return preds, pred_probs
    return pred_probs


def load_model(pretrain_model_dir, use_nsp, use_distill, num_labels=2):
    logger.info("load pretrained model from {}".format(pretrain_model_dir))
    if use_nsp:
        model = BertForNextSentencePrediction.from_pretrained(pretrain_model_dir)
        logger.info("use next sentence prediction model")
#     elif use_distill:
#         loss_fn = clf_distill_loss_functions.Plain()
#         model = BertDistill.from_pretrained(pretrain_model_dir, loss_fn=loss_fn)
#         logger.info("use distill model")
    else:
        model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=True)
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser()
    
    def str2bool(bstr):
        if bstr.lower() == 'true':
            return True
        return False
 
    ## Required parameters
    parser.add_argument("--pretrain_model_dir", default=r"bert-base-uncased", type=str)
    parser.add_argument("--thred", type=float, default=0.5)
    parser.add_argument("--use_nsp", type=str2bool, default=False)
    parser.add_argument("--use_distill", type=str2bool, default=False)
    parser.add_argument("--reverse", type=str2bool, default=False)
    parser.add_argument("--random_input", type=str2bool, default=False)
    parser.add_argument("--label_single", type=str2bool, default=True)
    parser.add_argument("--output_dir", default=None, type=str) 
    parser.add_argument("--input_fn", default=None, type=str)
    parser.add_argument("--label_fn", default=None, type=str)
    
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
    if args.random_input:
        rand_fn = os.path.join(args.output_dir, "rand.txt")
        random_text_fn(args.input_fn, rand_fn)
        args.input_fn = rand_fn

    argsDict = args.__dict__
    with open(os.path.join(args.output_dir, "setting.txt"), "w", encoding="utf-8") as fp:
        for eachArg, value in argsDict.items():
            fp.writelines(eachArg + ' : ' + str(value) + '\n')
              
    processor = NSPData(args.label_fn, input_fn=args.input_fn, reverse=args.reverse)

    label_list = processor.get_labels()
    num_labels = len(label_list)
    # Prepare model
    model, tokenizer = load_model(args.pretrain_model_dir, args.use_nsp, args.use_distill, num_labels)
    
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    '''load test set'''
    test_examples, test_label_list, test_hypo_type_list = processor.get_examples(label_single=args.label_single)
    print(test_hypo_type_list)
    
#     lines = ["\t".join(['index', 'sentence1', 'sentence2', 'label']) + "\n"]
#     for i, ex in enumerate(test_examples):
#         sent1 = ex.text_a.replace("\t", " ")
#         sent2 = ex.text_b.replace("\t", " ")
#         label = 'entailment' if ex.label == 'isNext' else 'not_entailment'
#         lines.append('\t'.join([str(i), sent1, sent2, label]) + "\n")
#     with open(os.path.join(args.output_dir, "test.tsv"), mode="w", encoding="utf-8") as fp:
#         fp.writelines(lines)
    
    test_features, _ = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)


    logits, pred_probs = eval(model, test_dataloader, device, num_labels, return_logits=True)
#     np.savetxt(os.path.join(args.output_dir, "final_test_logits.txt"), logits, delimiter=',')
#     np.savetxt(os.path.join(args.output_dir, "final_test_log.txt"), pred_probs, delimiter=',')
    

    text_num = len(test_label_list)

    if ("topic" in args.input_fn) or ("sst" in args.input_fn) or ("agnews" in args.input_fn) or ("snips" in args.input_fn):
        #single label, without none
        test_pred_label_list = predict_single(pred_probs, text_num, test_hypo_type_list, thred=-100)
        acc = cal_acc(test_label_list, test_pred_label_list)
        print(acc)
        with open(os.path.join(args.output_dir, "metric.txt"), "w", encoding="utf-8") as fp:
            fp.writelines(["accuracy: {}\n".format(acc)])  
    elif "emotion" in args.input_fn:
        #single label, with none
        if args.use_nsp:
            test_pred_label_list = predict_single(pred_probs, text_num, test_hypo_type_list, thred=args.thred)  
            wf1 = cal_wf1(test_pred_label_list, test_label_list, processor.classes)
            print(wf1)        
        else:
            test_pred_label_list = predict_single(pred_probs, text_num, test_hypo_type_list, thred=0.5)  
            wf1 = cal_wf1(test_pred_label_list, test_label_list, processor.classes)
            print(wf1)
        with open(os.path.join(args.output_dir, "metric.txt"), "w", encoding="utf-8") as fp:
            fp.writelines(["weighted f1: {}\n".format(wf1)])  
    elif "situation" in args.input_fn:
        if args.use_nsp:
            test_pred_label_list = predict_multi(pred_probs, text_num, test_hypo_type_list, thred=args.thred)
#             test_pred_label_list = predict_multi_baseline(pred_probs, text_num, test_hypo_type_list)
            wf1 = cal_wf1(test_pred_label_list, test_label_list, processor.classes)
            print(wf1)
        else:
            test_pred_label_list = predict_multi(pred_probs, text_num, test_hypo_type_list, thred=0.5)
            wf1 = cal_wf1(test_pred_label_list, test_label_list, processor.classes)
            print(wf1)
        with open(os.path.join(args.output_dir, "metric.txt"), "w", encoding="utf-8") as fp:
            fp.writelines(["weighted f1: {}\n".format(wf1)])    
    with open(os.path.join(args.output_dir, "final_test_pred.txt"), mode="w", encoding="utf-8") as fp:
        if isinstance(test_label_list[0], list):
            fp.writelines(["\t".join(x) + "\n" for x in test_pred_label_list])
        else:
            fp.writelines([x + "\n" for x in test_pred_label_list])
    
    print(args.pretrain_model_dir)
    



if __name__ == "__main__":
    main()