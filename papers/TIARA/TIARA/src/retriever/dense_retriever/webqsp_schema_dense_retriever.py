# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
from dataloader.webqsp_json_loader import WebQSPJsonLoader
from utils.config import webqsp_ptrain_path, webqsp_pdev_path, webqsp_test_path, webqsp_question_2hop_relation_path
import argparse
import json
from utils.metrics import Metrics
import os.path
import random
import time

import numpy as np
import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, IntervalStrategy

from retriever.freebase_retriever import FreebaseRetriever
from utils.file_util import pickle_load, read_json_file
from utils.hugging_face_dataset import HFDataset


class WebQSPSchemaDenseRetriever:
    def __init__(self, params):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_negative = params.get('num_negative', 20)
        random.seed(429)
        self.negative_strategy = params.get('negative_strategy', 'relation_sample')

        self.question_relations_map = pickle_load(webqsp_question_2hop_relation_path)

        self.webqsp_ptrain = WebQSPJsonLoader(webqsp_ptrain_path)
        self.webqsp_pdev = WebQSPJsonLoader(webqsp_pdev_path)
        self.webqsp_test = WebQSPJsonLoader(webqsp_test_path)
        self.retriever = FreebaseRetriever()
        print('negative strategy:', self.negative_strategy)

    def encode_relation(self, dataloader, output_dir):
        encoding_path = output_dir + '_relation_encodings'
        labels_path = output_dir + '_relation_labels'
        if self.negative_strategy != 'bootstrapping' and output_dir is not None and os.path.isdir(output_dir):
            if os.path.isfile(encoding_path) and os.path.isfile(labels_path):
                return torch.load(encoding_path), torch.load(labels_path)

        text_a = []  # question
        text_b = []  # relation
        labels = []  # label 1 or 0

        start_time = time.time()
        prediction_path = '../logs/' + 'webqsp_' + dataloader.get_dataset_split() + '_relations.json'
        predictions = None
        if os.path.isfile(prediction_path):
            predictions = read_json_file(prediction_path)

        for idx in tqdm(range(0, dataloader.len)):
            qid = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            golden_relations = dataloader.get_question_predicates_by_idx(idx)
            golden_reverse_relations = self.retriever.reverse_relation_list(golden_relations)

            # 1. positive samples
            for r in golden_relations:
                text_a.append(question)
                text_b.append(r)
                labels.append(1)

            # 2. negative samples
            candidates = list(self.question_relations_map[qid])
            negative_relations = set()
            if self.negative_strategy == 'question_sample':
                negative_idx = random.sample(range(0, dataloader.len), min(self.num_negative, dataloader.len))
                for neg_idx in negative_idx:
                    if neg_idx == idx:
                        continue
                    neg = dataloader.get_question_predicates_by_idx(neg_idx)
                    for r in neg:
                        if r in candidates and r not in golden_relations and r not in golden_reverse_relations:
                            negative_relations.add(r)
            elif self.negative_strategy == 'relation_sample':
                samples = random.sample(candidates, min(self.num_negative, len(candidates)))
                for r in samples:
                    if r not in golden_relations and r not in golden_reverse_relations:
                        negative_relations.add(r)
            elif self.negative_strategy == 'bootstrapping':
                assert predictions is not None
                for r in predictions[qid]['relations'][:self.num_negative]:
                    if r in candidates and r not in golden_relations and r not in golden_reverse_relations:
                        negative_relations.add(r)
            elif self.negative_strategy == 'all':
                for r in candidates:
                    if r not in golden_relations and r not in golden_reverse_relations:
                        negative_relations.add(r)

            for r in negative_relations:
                text_a.append(question)
                text_b.append(r)
                labels.append(0)

        print('avg len:', len(text_a) / dataloader.get_len())
        encodings = self.tokenizer(text_a, text_b, max_length=64, truncation=True, padding=True, return_tensors='pt')
        torch.save(encodings, encoding_path)
        torch.save(labels, labels_path)
        return encodings, labels

    def train(self, output_dir='../model/webqsp_schema_dense_retriever'):
        if self.negative_strategy != 'bootstrapping' and os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('[INFO] Model already exists, skip training')
            return

        train_encodings, train_labels = self.encode_relation(self.webqsp_ptrain, output_dir=output_dir + '/train')
        dev_encodings, dev_labels = self.encode_relation(self.webqsp_pdev, output_dir=output_dir + '/dev')

        train_dataset = HFDataset(train_encodings, train_labels)
        dev_dataset = HFDataset(dev_encodings, dev_labels)

        # training settings
        metric = load_metric("accuracy")
        if self.negative_strategy != 'bootstrapping':
            model_name = self.model_name
        else:
            model_name = output_dir
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.train()

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                          per_device_train_batch_size=128, per_device_eval_batch_size=100, num_train_epochs=3, learning_rate=3e-5,
                                          evaluation_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.EPOCH, load_best_model_at_end=True)
        trainer = Trainer(self.model, args=training_args if self.negative_strategy != 'bootstrapping' else None,
                          train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_metrics)
        best_run = trainer.train(resume_from_checkpoint=output_dir if self.negative_strategy == 'bootstrapping' else None)
        trainer.save_model(output_dir)

    def predict(self, dataloader, output_dir):
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.model.eval()
        self.model.to(self.device)

        chunk_size = 500
        metrics = Metrics()
        top = [1, 5, 10, 15, 20, 30, 100]

        predictions = dict()
        for idx in tqdm(range(dataloader.len)):
            qid = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            golden_schema = dataloader.get_question_predicates_by_idx(idx)

            candidates = list(self.question_relations_map[qid])
            candidate_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
            scores = []
            for chunk_idx in range(0, len(candidate_chunks)):
                try:
                    score = self.classify_question_schema(question, candidate_chunks[chunk_idx])
                    scores += list(score)
                except Exception as e:
                    print('exception' + str(e))

            scores = np.array(scores)
            predicted_idx = (-scores).argsort()[:100]

            hits = dict()
            predicted_schema = [candidates[i] for i in predicted_idx[:100]]
            for k in range(0, min(100, len(predicted_schema))):
                if predicted_schema[k] in golden_schema:
                    for t in top:
                        if k < t:
                            hits[t] = 1

            for t in top:
                metrics.add_metric('hits@' + str(t), hits.get(t, 0))
            metrics.count()
            predictions[qid] = {'question': question, 'relations': predicted_schema[:50]}
        # end for each question
        print(metrics.get_metrics(['hits@1', 'hits@5', 'hits@10', 'hits@15', 'hits@20', 'hits@30', 'hits@100']))

        with open('../dataset/WebQSP/schema_linking_results/' + 'webqsp_' + dataloader.get_dataset_split() + '_relations.json', 'w') as f:
            json.dump(predictions, f)

    def classify_question_schema(self, question: str, schema: list):
        text_a = [question for _ in range(0, len(schema))]
        text_b = schema
        try:
            encodings = self.tokenizer(text_a, text_b, max_length=64, truncation=True, padding=True, return_tensors='pt')
            predictions = self.model(input_ids=encodings['input_ids'].to(self.device), attention_mask=encodings['attention_mask'].to(self.device),
                                     token_type_ids=encodings['token_type_ids'].to(self.device))
            scores = predictions.logits.detach().cpu().numpy()
            return scores[:, 1]
        except Exception as e:
            print('classify_question_schema: ' + str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg', type=str, default='relation_sample', choices=['relation_sample', 'question_sample', 'bootstrapping'])
    parser.add_argument('--num', type=int, default=30)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    params = {'negative_strategy': args.neg, 'num_negative': args.num}
    schema_dense_retriever = WebQSPSchemaDenseRetriever(params)
    schema_dense_retriever.train()

    if args.split == 'train':
        schema_dense_retriever.predict(schema_dense_retriever.webqsp_ptrain, '../model/webqsp_schema_dense_retriever')
    elif args.split == 'dev':
        schema_dense_retriever.predict(schema_dense_retriever.webqsp_pdev, '../model/webqsp_schema_dense_retriever')
    elif args.split == 'test':
        schema_dense_retriever.predict(schema_dense_retriever.webqsp_test, '../model/webqsp_schema_dense_retriever')
