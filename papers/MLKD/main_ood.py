# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import os
import sys
import time

import datasets
import torch

from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    # CLM perplexity
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,

    # PretrainedConfig,
    SchedulerType,
    default_data_collator,
    set_seed
)
from transformers.utils.versions import require_version

from utils import data_utils

from models.clm import run_CLM
from models.kd_clm import run_KD_CLM
from models.kd_clm_v1 import run_KD_CLM as run_KD_CLM_v1

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def setup_logger(args, level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    log_name = "{}/log-{}-{}-{}".format(args.output_dir, args.method, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.log_suffix)
    fh = logging.FileHandler(log_name)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    # task & data
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--cache_dir_data", type=str, default='./dataset/cache')
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels of the data to train the feature extractor.")
    # model & training
    parser.add_argument("--method", type=str, help="Method used for OOD detection.",
                        choices=["MSP", "CLM", "MLM", "MDF", "IDLM_MDF", "BCAD_MDF", "KD_CLM"])
    parser.add_argument("--model_name_or_path", default="gpt2", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--max_length", type=int, default=128,
        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--batch_size_train", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--batch_size_eval", type=int, default=16, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--seed", type=int, default=667, help="A seed for reproducible training.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", help="ids of the gpus to use")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X update steps.")
    parser.add_argument("--log_suffix", type=str, default='', help="suffix to append to the name of the log file.")
    parser.add_argument("--stu_id", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--tea_id", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--loop_type", type=str, default='kl', help="suffix to append to the name of the log file.")
    parser.add_argument("--kd_teacher_dir", type=str, default='./results/CLM_5', help="suffix to append to the name of the log file.")
    parser.add_argument("--intermediate_mode", type=str, default='none', help="suffix to append to the name of the log file.")
    parser.add_argument("--train_from_scratch", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--only_eval", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--temperature", type=float, default=1.0, help="kd temperature")
    parser.add_argument("--use_mse_loss", type=str, default="none", help="The mode of using mse loss.")

    args = parser.parse_args()

    args.do_train = True if args.train_file and not args.only_eval else False
    args.do_eval = True if args.validation_file else False
    args.do_test = True if args.test_file else False
    args.output_dir = f"{args.output_dir}/{args.seed}"
    if args.method == 'KD_CLM' and is_rel(args):
        args.use_mse_loss = "both"
    if is_comp(args):
        args.num_train_epochs = 5

    return args

def is_rel(args):
    return "20ng/train/rel" in args.train_file

def is_comp(args):
    return "20ng/train/comp" in args.train_file

def main():
    args = parse_args()

    ## Setup logging, random seed
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_logger(args, level=logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.seed is not None:
        set_seed(args.seed)

    # Setup CUDA, GPU & distributed training
    args.n_gpu = len(args.gpu_ids)  # torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids[0])
    device = torch.device("cuda")
    args.device = device

    # Load Config, Tokenizer, Model
    if not args.do_train:
        args.model_name_or_path = args.output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if tokenizer.pad_token is None:
        logger.info('Padding token is None! Take `eos_token` as `pad_token`.')
        tokenizer.pad_token = tokenizer.eos_token

    if args.method == 'MSP':
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config).to(device)
    elif args.method == 'CLM':
        if args.only_eval:
            args.batch_size_train = 1
        args.pad_to_max_length = True
        args.batch_size_eval = 1 # to ease the computation of perplexity.
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        if args.train_from_scratch:
            # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config).to(device)
            logger.info("Load new model from scratch...")
            model = AutoModelForCausalLM.from_config(config).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config).to(device)
        data_collator = default_data_collator
    elif args.method =='KD_CLM':
        args.pad_to_max_length = True
        args.batch_size_eval = 1 # to ease the computation of perplexity.
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        if args.train_from_scratch:
            logger.info("Load new model from scratch")
            student = AutoModelForCausalLM.from_config(config).to(device)
        else:
            student = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config).to(device)
        data_collator = default_data_collator
        # load teacher model
        config_teacher = AutoConfig.from_pretrained(f'{args.kd_teacher_dir}/{args.seed}')
        teacher = AutoModelForCausalLM.from_pretrained(f'{args.kd_teacher_dir}/{args.seed}', config=config_teacher).to(device)
    elif args.method == 'MLM':
        args.batch_size_eval = 1 # to ease the computation of perplexity.
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config).to(device)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    elif args.method == 'MDF':
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModel.from_pretrained(args.model_name_or_path, config=config).to(device)
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        raise ValueError(f'Invalid method [{args.method}]')
        
    ## Load dataset
    padding = "max_length" if args.pad_to_max_length else False
    tokenized_datasets, label_list = data_utils.load_dataset_and_tokenize(tokenizer, args.train_file, args.validation_file, args.test_file, args.max_length, padding, args.cache_dir_data, args.seed)
    label_list = None
    if label_list is not None and len(label_list) != args.num_labels:
        raise ValueError('Numbers of labels do not match!')
    
    if args.method == 'MSP':
        # Set the correspondences label/ID inside the model config
        if len(model.config.label2id) != len(label_list):
            raise ValueError('Error occurs in label list consistency! !')
        else:
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {i: l for i, l in enumerate(label_list)}

    ## DataLoaders creation:
    data_loader = {}
    for k, d in tokenized_datasets.items():
        shuffle, batch_size = (True, args.batch_size_train) if k == 'train' else (False, args.batch_size_eval)
        if isinstance(d, list):
            data_loader[k] = [DataLoader(ii, shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size) for ii in d]
        else:
            data_loader[k] = DataLoader(d, shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)

    if args.method == 'CLM':
        run_CLM(logger, args, model, data_loader, tokenized_datasets, tokenizer)
    if args.method == 'KD_CLM':
        if is_rel(args):
            run_KD_CLM_v1(logger, args, teacher, student, data_loader, tokenized_datasets, tokenizer, temperature=args.temperature, use_mse_loss=args.use_mse_loss)
        else:
            run_KD_CLM(logger, args, teacher, student, data_loader, tokenized_datasets, tokenizer, temperature=args.temperature, use_mse_loss=args.use_mse_loss)


if __name__ == "__main__":
    sys.argv += [
        '--batch_size_train', '8',
        ]

    main()