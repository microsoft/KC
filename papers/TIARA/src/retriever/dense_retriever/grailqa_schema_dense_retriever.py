# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')

import os
from tqdm import tqdm
import argparse
import json
from utils.metrics import Metrics
import os.path
import random

import numpy as np
import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, IntervalStrategy

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import grailqa_train_path, grailqa_dev_path, grailqa_test_path
from utils.file_util import read_list_file
from utils.hugging_face_dataset import HFDataset


class GrailQASchemaDenseRetriever:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_negative = 20
        self.negative_strategy = 'random_question'
        random.seed(429)

        self.all_class = read_list_file('../dataset/grail_classes.txt')
        self.all_relation = read_list_file('../dataset/grail_relations.txt')

        self.grailqa_train = GrailQAJsonLoader(grailqa_train_path)
        self.grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)
        self.grailqa_test = GrailQAJsonLoader(grailqa_test_path)
        self.retriever = FreebaseRetriever()

    def encode(self, dataloader, output_dir, schema_type):
        encoding_path = output_dir + '_' + schema_type + '_encodings'
        labels_path = output_dir + '_' + schema_type + '_labels'
        if output_dir is not None:
            if os.path.isfile(encoding_path) and os.path.isfile(labels_path):
                return torch.load(encoding_path), torch.load(labels_path)

        text_a = []  # question
        text_b = []  # schema
        labels = []

        for idx in tqdm(range(0, dataloader.len)):
            question = dataloader.get_question_by_idx(idx)
            golden_class = dataloader.get_golden_class_by_idx(idx, only_label=True)
            golden_relation = dataloader.get_golden_relation_by_idx(idx)
            golden_reverse_relation = self.retriever.reverse_relation_list(golden_relation)

            # 1. positive samples
            golden_schema = golden_class if schema_type == 'class' else golden_relation
            for s in golden_schema:
                text_a.append(question)
                text_b.append(s)
                labels.append(1)

            # 2. negative samples
            negative_class = []
            negative_relation = []
            if self.negative_strategy == 'random':
                negative_class = random.sample(self.all_class, self.num_negative)
                negative_relation = random.sample(self.all_relation, self.num_negative)
            elif self.negative_strategy == 'random_question':
                for i in range(self.num_negative):
                    random_idx = random.randrange(0, dataloader.len)
                    while random_idx == idx:
                        random_idx = random.randrange(0, dataloader.len)
                    negative_class.extend(dataloader.get_golden_class_by_idx(random_idx, True))
                    negative_relation.extend(dataloader.get_golden_relation_by_idx(random_idx))

            negative_schema = negative_class if schema_type == 'class' else negative_relation
            for s in negative_schema:
                if s not in golden_schema and s not in golden_reverse_relation:
                    text_a.append(question)
                    text_b.append(s)
                    labels.append(0)
        # end for each question

        encodings = self.tokenizer(text_a, text_b, padding=True, truncation=True, max_length=128, return_tensors='pt')
        torch.save(encodings, encoding_path)
        torch.save(labels, labels_path)
        return encodings, labels

    def train(self, schema_type, output_dir='../model/schema_dense_retriever'):
        output_dir = output_dir + '/' + schema_type
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('[INFO] Model already exists, skip training')
            return

        train_encodings, train_labels = self.encode(self.grailqa_train, output_dir=output_dir + '/train', schema_type=schema_type)
        dev_encodings, dev_labels = self.encode(self.grailqa_dev, output_dir=output_dir + '/dev', schema_type=schema_type)

        train_dataset = HFDataset(train_encodings, train_labels)
        dev_dataset = HFDataset(dev_encodings, dev_labels)

        # training settings
        metric = load_metric("accuracy")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.train()

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                          per_device_train_batch_size=256, per_device_eval_batch_size=64, num_train_epochs=3, learning_rate=5e-5,
                                          evaluation_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.EPOCH, load_best_model_at_end=True)
        trainer = Trainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_metrics)
        best_run = trainer.train()
        trainer.save_model(output_dir)

    def predict(self, dataloader, output_dir, schema_type):
        output_dir = output_dir + '/' + schema_type
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.model.eval()
        self.model.to(self.device)

        chunk_size = 500
        if schema_type == 'class':
            all_schema_chunks = [self.all_class[i:i + chunk_size] for i in range(0, len(self.all_class), chunk_size)]
        else:
            all_schema_chunks = [self.all_relation[i:i + chunk_size] for i in range(0, len(self.all_relation), chunk_size)]
        assert len(all_schema_chunks)

        metrics = Metrics()
        top = [1, 5, 10, 15, 20, 30, 100]

        predictions = dict()
        level_count = dict()
        for idx in tqdm(range(0, dataloader.get_len())):
            qid = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            golden_schema = level = None
            if dataloader.get_dataset_split() != 'test':
                level = dataloader.get_level_by_idx(idx)
                golden_schema = dataloader.get_golden_class_by_idx(idx, True) if schema_type == 'class' else dataloader.get_golden_relation_by_idx(idx)

            scores = []
            for chunk_idx in range(0, len(all_schema_chunks)):
                try:
                    score = self.classify_question_schema(question, all_schema_chunks[chunk_idx])
                    scores += list(score)
                except Exception as e:
                    print('exception' + str(e))

            scores = np.array(scores)
            predicted_idx = (-scores).argsort()[:100]

            if schema_type == 'class':
                predicted_schema = [self.all_class[i] for i in predicted_idx[:100]]
            else:
                predicted_schema = [self.all_relation[i] for i in predicted_idx[:100]]

            if dataloader.get_dataset_split() != 'test':
                hits = dict()
                for k in range(0, 100):
                    if predicted_schema[k] in golden_schema:
                        for t in top:
                            if k < t:
                                hits[t] = 1

                for t in top:
                    if t in hits:
                        metrics.add_metric('hits@' + str(t), hits[t])
                        level_count['hits@' + str(t) + '_' + level] = level_count.get('hits@' + str(t) + '_' + level, 0) + 1
                    else:
                        metrics.add_metric('hits@' + str(t), 0)

                metrics.count()
                level_count[level] = level_count.get(level, 0) + 1

                print('[' + str(idx) + '] ')
                print(metrics.get_metrics(['hits@' + str(k) for k in top]))
                for l in ['i.i.d.', 'compositional', 'zero-shot']:
                    for k in top:
                        print('[' + l + ']' + ' hits@' + str(k) + ': ' + str(level_count.get('hits@' + str(k) + '_' + l, 0) / level_count.get(l, 1)))
                print()

            schema_key = 'classes' if schema_type == 'class' else 'relations'
            predictions[qid] = {'question': question, schema_key: predicted_schema[:10]}
        # end for each question

        with open('../logs/grailqa_' + dataloader.get_dataset_split() + '_' + schema_type + '_predictions.json', 'w') as f:
            json.dump(predictions, f)

    def classify_question_schema(self, question: str, schema: list):
        text_a = [question for _ in range(0, len(schema))]
        text_b = schema
        try:
            encodings = self.tokenizer(text_a, text_b, max_length=128, truncation=True, padding=True, return_tensors='pt')
            predictions = self.model(input_ids=encodings['input_ids'].to(self.device), attention_mask=encodings['attention_mask'].to(self.device),
                                     token_type_ids=encodings['token_type_ids'].to(self.device))
            scores = predictions.logits.detach().cpu().numpy()
            return scores[:, 1]
        except Exception as e:
            print('classify_question_schema: ' + str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema_type', choices=['class', 'relation'], required=True)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    args = parser.parse_args()

    schema_dense_retriever = GrailQASchemaDenseRetriever()
    schema_dense_retriever.train(args.schema_type)
    split = None
    if args.split == 'train':
        split = schema_dense_retriever.grailqa_train
    elif args.split == 'dev':
        split = schema_dense_retriever.grailqa_dev
    elif args.split == 'test':
        split = schema_dense_retriever.grailqa_test
    schema_dense_retriever.predict(split, '../model/schema_dense_retriever', args.schema_type)
