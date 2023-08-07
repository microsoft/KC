#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os 
import torch 
import logging 
import argparse
import numpy as np 
import pandas as pd 
import warnings
from ood_main import load_dataset
warnings.filterwarnings('ignore')

from ood_main import load_extra_dataset

from simpletransformers.classification import  ClassificationModel 
from simpletransformers.language_modeling import LanguageModelingModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

from ood_metrics import compute_all_scores
from sklearn.model_selection import train_test_split

seed=42

def set_binary_label(dataframe, indomain, col='labels'):
    if indomain:
        dataframe[col].values[:] = 1
    else:
        dataframe[col].values[:] = 0

def train_process(args):
    # load data
    data_type = args.data_type.strip()
    print("data type %s"%( data_type))
    train_df, num_classes = load_dataset('clinc150_train', data_type=data_type)
    test_df, _ = load_dataset('clinc150_test', data_type=data_type)
    eval_df, _ = load_dataset('clinc150_val', data_type=data_type)

    if args.model_class == 'bert':
        model = ClassificationModel(
        'bert', 
        'bert-base-uncased', 
        num_labels=num_classes,
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 5,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'learning_rate': 4e-5,
              'warmup_ratio': 0.10,
              'train_batch_size': 32,
              'eval_batch_size': 32, 
              'evaluate_during_training': False,
              'evaluate_during_training_steps': 2000,    
              'do_lower_case': True,
              'reprocess_input_data': True, 
              'overwrite_output_dir': False,
              'output_dir': './models/%s_outputs/'%(data_type),
              'best_model_dir': "./models/%s_outputs/best_model"%(data_type),
              'cache_dir': "./models/%s_cache_dir/"%(data_type)})
    else:
        raise NotImplementedError

    model.train_model(train_df, eval_df=eval_df)
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    # pdb.set_trace()
    print(result)
    print("---------------")

def finetune_bcad(args):
    # load data
    sed = 42
    data_type = args.data_type.strip()
    print("data type %s"%( data_type))
    if args.load_path:
        output_dir = './models/{}_{}_ft_MLM_binary_intent_outputs_{}/'.format(args.model_class, data_type, str(args.neg_sample))
    else:
        output_dir = './models/{}_{}_ft_binary_intent_outputs_{}/'.format(args.model_class, data_type, str(args.neg_sample))
    print (output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if data_type == "sst":
        train_df = load_extra_dataset("./dataset/sst/sst-train.txt")
        eval_df = load_extra_dataset("./dataset/sst/sst-dev.txt")
        test_df = load_extra_dataset("./dataset/sst/sst-test.txt")
        for df in [train_df, test_df, eval_df]:
            set_binary_label(df, indomain=True) # set label to one (in domain)
    else:
        train_df = load_dataset('clinc150_train', data_type=data_type)
        test_df = load_dataset('clinc150_test', data_type=data_type)
        eval_df = load_dataset('clinc150_val', data_type=data_type)
        for df in [train_df, test_df, eval_df]:
            set_binary_label(df, indomain=True) # set label to one (in domain)
        # neg_df = load_extra_dataset("./dataset/sst-train.txt")
        # neg_val_df = load_extra_dataset("./dataset/sst-dev.txt")
        # neg_test_df = load_extra_dataset("./dataset/sst-test.txt")
    print ("training df size", len(train_df))
    book_df = load_extra_dataset("./dataset/bookcorpus/subset_books.txt")
    set_binary_label(book_df, indomain=False)
    wiki_df = load_extra_dataset("./dataset/wikipedia/squad_train_wiki.txt")
    set_binary_label(wiki_df, indomain=False)
    # pdb.set_trace()
    # train_df = train_df.sample(n=200, random_state=seed)
    neg_book_df = book_df.sample(n=args.neg_sample, random_state=seed)
    neg_book_val_df = book_df.sample(n=1000, random_state=seed)
    neg_book_test_df = book_df.sample(n=2000, random_state=seed)
    neg_wiki_df = wiki_df.sample(n=args.neg_sample, random_state=seed)
    neg_wiki_val_df = wiki_df.sample(n=1000, random_state=seed)
    neg_wiki_test_df = wiki_df.sample(n=2000, random_state=seed)

    if data_type != "sst":
        neg_book_df['text'] = neg_book_df['text'].apply(lambda x: x.strip(".? "))
        neg_book_val_df['text'] = neg_book_val_df['text'].apply(lambda x: x.strip(".? "))
        neg_book_test_df['text'] = neg_book_test_df['text'].apply(lambda x: x.strip(".? "))
        neg_wiki_df['text'] = neg_wiki_df['text'].apply(lambda x: x.strip(".? "))
        neg_wiki_val_df['text'] = neg_wiki_val_df['text'].apply(lambda x: x.strip(".? "))
        neg_wiki_test_df['text'] = neg_wiki_test_df['text'].apply(lambda x: x.strip(".? "))

    train_df = pd.concat([train_df, neg_book_df, neg_wiki_df])
    val_df = pd.concat([eval_df, neg_book_val_df, neg_wiki_val_df])
    
    print ("train_df", len(train_df['labels']), train_df['labels'][10:20], train_df['text'][0:10])
    if args.load_path:
        load_path = args.load_path
    else:
        load_path = 'bert-base-uncased' if args.model_class == 'bert' else 'roberta-base'
    if args.model_class == 'bert' or args.model_class == 'roberta':
        model = ClassificationModel(
        args.model_class, 
        load_path, 
        num_labels=2,
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 2,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'learning_rate': 4e-5,
              'warmup_ratio': 0.10,
              'do_lower_case': True,
              "max_seq_length": 256,
              "train_batch_size": 16,
              'reprocess_input_data': True, 
              'overwrite_output_dir': True,
              'save_model_every_epoch': False,
              'evaluate_during_training': True,
              'evaluate_during_training_verbose': True,
              'evaluate_during_training_steps': 2000,   
              'output_dir': output_dir,
              'best_model_dir': "{}/best_model".format(output_dir),
              'cache_dir': output_dir.replace("outputs", "cache_dir")})
    else:
        raise NotImplementedError

    # model.train_model(pd.concat([train_df, neg_df]), eval_df=pd.concat([eval_df, neg_val_df]))
    model.train_model(train_df, eval_df=eval_df)
    # result, model_outputs, wrong_predictions = model.eval_model(pd.concat([test_df, neg_test_df]))
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)    
    # pdb.set_trace()
    print(result)
    print (output_dir)
    print("---------------")

def load_ROSTD_extra_dataset(file_path="./dataset/SSTSentences.txt", drop_index=False, label=0):
    with open(file_path, 'r') as f:
        data = [ii.strip() for ii in f.readlines()]
    df = pd.DataFrame(data, columns=['text'])
    df["index"] = df.index
    df['labels'] = label
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df


def finetune_imlm(args):
    # load data
    data_type = args.data_type.strip()
    print("data type %s"%( data_type))
    output_dir = './models/{}_{}_mlm_ft_outputs_{}/'.format(args.model_class, data_type, seed)
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.txt')
    eval_file = os.path.join(output_dir, 'eval.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    if not (os.path.exists(train_file) and os.path.exists(eval_file)):
        if data_type == "sst":
            train_df = load_extra_dataset("./dataset/sst/sst-train.txt", label=1)
            eval_df = load_extra_dataset("./dataset/sst/sst-dev.txt", label=1)
            # test_df = load_extra_dataset("./dataset/sst-test.txt", label=1)
        elif data_type == "ROSTD":
            train_file = f"./HC3/hc3_train.txt"
            train_df = load_ROSTD_extra_dataset(train_file, label=1)
            train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=seed)
        else:
            train_df = load_dataset('clinc150_train', data_type=data_type)
            # test_df = load_dataset('clinc150_test', data_type=data_type)
            eval_df  = load_dataset('clinc150_val', data_type=data_type)
        # pdb.set_trace()
        print ("number of training instances", len(train_df['text']))
        with open(train_file, "w") as f:
            for line in train_df['text']:
                f.write(line + "\n")
        
        print ("number of eval instances", len(eval_df['text']))
        with open(eval_file, "w") as f:
            for line in eval_df['text']:
                f.write(line + "\n")

    if args.model_class == 'bert' or args.model_class == 'roberta':
        model = LanguageModelingModel(
        args.model_class, 
        'bert-base-uncased' if args.model_class == 'bert' else 'roberta-base',
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 10,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'do_lower_case': True,
              "max_seq_length": 128,
              "train_batch_size": 4,
              'reprocess_input_data': True, 
              'overwrite_output_dir': True,
              'save_model_every_epoch': False,
              'evaluate_during_training': True,
              'output_dir': output_dir,
              'best_model_dir': "{}/best_model".format(output_dir),
              'cache_dir': output_dir.replace("outputs", "cache_dir")})
    else:
        raise NotImplementedError
    model.train_model(train_file, eval_file=eval_file)
    model.eval_model(eval_file)
    # print(result)
    print("---------------")

def main(args):
    global seed
    seed = args.seed
    run_type = args.type
    # set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(args.gpu_id)
    if run_type == 'train_classifier':
        train_process(args)
    elif run_type == 'finetune_bcad':
        finetune_bcad(args)
    elif run_type == 'finetune_imlm':
        finetune_imlm(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bert Model OOD Fine-tuning")
    parser.add_argument('--type', default='finetune_bcad', choices=['train_classifier', 'finetune_bcad', 'finetune_imlm'])
    parser.add_argument('--model_class', default='bert', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--n_rep', default=1, type=int)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--data_type', default='ROSTD', choices = ['clinic', 'sst', "ROSTD"])
    parser.add_argument('--neg_sample',  default=7500, type=int)
    parser.add_argument('--seed', default=171, type=int)

    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0 python finetune_bert.py --type finetune_imlm --data_type ROSTD --model_class bert