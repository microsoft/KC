"""
Script to train BERT on MNLI with our loss function

Modified from the script from
https://github.com/UKPLab/emnlp2020-debiasing-unknown/blob/main/src/train_distill_bert.py

License: Apache License 2.0
"""

import argparse
import json
import logging
import os
import random
from os.path import join, exists
from typing import List, Dict, Iterable

import sys

import numpy as np

from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_bert import BertTokenizer

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from tqdm import trange, tqdm


from bert_distill import BertDistill

import clf_distill_loss_functions

import utils
from utils import Processor, process_par
from load_data import *


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias


class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        for example in data:
            tokens_a = tokenizer.tokenize(example.premise)

            tokens_b = None
            if example.hypothesis:
                tokens_b = tokenizer.tokenize(example.hypothesis)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    bias=None
                ))
        return features


class InputFeatureDataset(Dataset):

    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(np.array([x.label_id for x in batch], np.int64))

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()

    if batch[0].bias is None:
        return example_ids, input_ids, mask, segment_ids, label_ids

    bias = torch.tensor([x.bias for x in batch])

    return example_ids, input_ids, mask, segment_ids, label_ids, bias


class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        if batch_size == 1:
            raise NotImplementedError()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + 601767 + self.seed)
        n_batches = len(self)
        batch_lens = np.full(n_batches, self.batch_size, np.int32)

        # Randomly select batches to reduce by size 1
        extra = n_batches * self.batch_size - len(self.data_source)
        batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

        batch_ends = np.cumsum(batch_lens)
        batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

        if batch_ends[-1] != len(self.data_source):
            print(batch_ends)
            raise RuntimeError()

        bounds = np.stack([batch_starts, batch_ends], 1)
        rng.shuffle(bounds)

        for s, e in bounds:
            yield np.arange(s, e)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
    if sorted:
        data.sort(key=lambda x: len(x.input_ids))
        ds = InputFeatureDataset(data)
        sampler = SortedBatchSampler(ds, batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_input_features)
    else:
        ds = InputFeatureDataset(data)
        return DataLoader(ds, batch_size=batch_size, sampler=RandomSampler(ds),
                          collate_fn=collate_input_features)


def build_eval_dataloader(data: List[InputFeatures], batch_size):
    ds = InputFeatureDataset(data)
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                      collate_fn=collate_input_features)


def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer)
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def main():
    parser = argparse.ArgumentParser()
    def str2bool(bstr):
        if bstr.lower() == 'true':
            return True
        return False
    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input directory (the MNLI data folder) which contains train, dev NLI datasets.")
    parser.add_argument("--hans_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The hans directory which contains heuristics_evaluation_set.txt")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--debug_num", type=int, default=2000)
    parser.add_argument("--custom_bias", default=None)
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        type=str2bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                         default=False,
                        type=str2bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_on_train",
                        default=False,
                        type=str2bool,
                        help="Whether to run eval on the train set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda",
                        default=False,
                        type=str2bool,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        type=str2bool,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    ## Our arguements
    parser.add_argument("--mode", choices=["none", "label_smoothing", 
                                           "reweight_baseline", "bias_product_baseline"])
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--sorted", type=str2bool, default=False,
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster. Our experiments did not actually'
                             ' use this in the end (not sure if it makes a difference) so '
                             'its off by default.')

    args = parser.parse_args()

    utils.add_stdout_logger()

    if args.mode == "none":
        loss_fn = clf_distill_loss_functions.Plain()
    elif args.mode == "label_smoothing":
        loss_fn = clf_distill_loss_functions.LabelSmoothing(len(NLI_LABELS))
    elif args.mode == "reweight_baseline":
        loss_fn = clf_distill_loss_functions.ReweightBaseline()
    elif args.mode == "bias_product_baseline":
        loss_fn = clf_distill_loss_functions.BiasProductBaseline()
    else:
        raise RuntimeError()

    output_dir = args.output_dir

    if args.do_train:
        if exists(output_dir):
            if len(os.listdir(output_dir)) > 0:
                logging.warning("Output dir exists and is non-empty")
        else:
            os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        logging.warning(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Its way ot easy to forget if this is being set by a command line flag
    if "-uncased" in args.bert_model:
        do_lower_case = True
    elif "-cased" in args.bert_model:
        do_lower_case = False
    else:
        raise NotImplementedError(args.bert_model)

    argsDict = args.__dict__
    with open(os.path.join(args.output_dir, "setting.txt"), "w", encoding="utf-8") as fp:
        for eachArg, value in argsDict.items():
            fp.writelines(eachArg + ' : ' + str(value) + '\n')
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=do_lower_case)

    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
        ####################################################
        train_examples = load_mnli(args, True, args.debug_num if args.debug else None)
        num_train_optimization_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        loss_fn.num_train_optimization_steps = int(num_train_optimization_steps)
        loss_fn.num_epochs = int(args.num_train_epochs)


    config = BertConfig.from_pretrained(
        args.bert_model,
        num_labels=len(NLI_LABELS),
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = BertDistill.from_pretrained(
        args.bert_model,
        from_tf=bool(".ckpt" in args.bert_model),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        loss_fn=loss_fn
    )
#     print(model.classifier.weight[:10,:])
#     print(model.classifier.bias)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps
        )
        
        train_features: List[InputFeatures] = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, args.n_processes)

        bias_map = load_bias(custom_path=args.custom_bias)
#         print(len(bias_map))
        for fe in train_features:
            if bias_map is None:
                fe.bias = None
            else:
                fe.bias = bias_map[fe.example_id].astype(np.float32)

        example_map = {}
        for ex in train_examples:
            example_map[ex.id] = ex

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size,
                                                  args.seed, args.sorted)

        model.train()
        model.zero_grad()
        total_steps = 0

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                if bias_map is not None:
                    example_ids, input_ids, input_mask, segment_ids, label_ids, bias = batch
                else:
                    bias = None
                    example_ids, input_ids, input_mask, segment_ids, label_ids = batch
                logits, loss = model(input_ids, segment_ids, input_mask, label_ids, bias)
                total_steps += 1

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        
        print(model_to_save.classifier.weight[:10,:])
        print(model_to_save.classifier.bias)
        
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

#         model = BertDistill.from_pretrained(
#             args.output_dir,
#             loss_fn=loss_fn
#         )
#         model.to(device)
    else:
        model = BertDistill.from_pretrained(
            args.output_dir,
            loss_fn=loss_fn
        ) 
        print(model.classifier.weight[:10,:])
        print(model.classifier.bias)
        model.to(device)

    if not args.do_eval:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
        eval_datasets = []
        if args.do_eval_on_train:
            eval_datasets += [("mnli_train", load_mnli(args, True))]
            eval_datasets += [("mnli_sp_train", load_mnli(args, False, custom_path="sample_train.tsv"))]
        else:
            eval_datasets += [("mnli_dev_m", load_mnli(args, False)),
                             ("mnli_dev_mm", load_mnli(args, False, custom_path="dev_mismatched.tsv"))]
            eval_datasets += [("hans", load_hans(args))]
    else:
        eval_datasets = []


    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_logits = []
        test_subm_ids = []

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, len(NLI_LABELS)), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_logits.append(logits.detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        probs = np.concatenate(probs, 0)
        test_logits = np.concatenate(test_logits, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(probs, axis=1)

        result = {"acc": simple_accuracy(preds, all_label_ids)}
        result["loss"] = eval_loss

        output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
        output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
            logging.info("***** Eval results *****")
            all_writer.write("eval results on %s:\n" % name)
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                all_writer.write("%s = %s\n" % (key, str(result[key])))
        #output probs
        output_answer_file = os.path.join(output_dir, "eval_%s_probs.json" % name)
        answers = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, probs)}
        with open(output_answer_file, "w") as f:
            json.dump(answers, f)
        #output logits
        output_logits_file = os.path.join(output_dir, "eval_%s_logits.json" % name)
        logits = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, test_logits)}
        with open(output_logits_file, "w") as f:
            json.dump(logits, f)
            

if __name__ == "__main__":
    main()
