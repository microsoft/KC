# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')

from retriever.ranking_candidate import LogicalFormRetriever
from utils.question_pattern import get_question_with_entity_and_schema, get_question_with_candidate_query

import os
from tqdm import tqdm
import argparse
import json
import os.path
import time
import numpy as np
import torch
from retriever.schema_linker.schema_file_reader import SchemaLinker
from algorithm.s_expr_logits_processor import SExpressionLogitsProcessor
from datasets import load_metric
from transformers import IntervalStrategy, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, LogitsProcessorList
from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import grailqa_train_path, grailqa_dev_path, grailqa_test_path, grailqa_generation_model_path, freebase_addr, freebase_port, input_path, grailqa_fb_types_path, \
    grailqa_schema_train_path, grailqa_schema_dev_path, grailqa_schema_test_path, grailqa_rng_ranking_train_path, \
    output_path, grailqa_tiara_ranking_dev_path, grailqa_tiara_dev_el_path, grailqa_tiara_test_el_path, grailqa_tiara_ranking_test_path
from retriever.entity_linker.grailqa_entity_linker import GrailQAEntityLinker
from utils.file_util import pickle_load, pickle_save
from utils.hugging_face_dataset import HFDataset2
from utils.log_util import save_log
from utils.logic_form_util import lisp_to_sparql
from utils.metrics import Metrics, get_precision, get_recall, get_f1_score_by_pr
from utils.string_util import string_to_bool, is_schema_in_lisp, get_entity_schema_in_lisp


class GrailQAGeneration:

    def __init__(self, params: dict, mode='qa'):
        self.params = params
        self.prompt = params.get('prompt', 'lf_schema')
        self.checker = string_to_bool(params.get('checker', 'True'))
        if 'lf' not in self.prompt:  # without logical form retrieval
            self.back_to_rank = False
        else:
            self.back_to_rank = params.get('back_to_rank', True)
        self.grailqa_entity_linker = GrailQAEntityLinker(params['grailqa_dev_el'], params['grailqa_train'], test_file_path=params.get('grailqa_test_el', None))
        self.grailqa_schema_linker = SchemaLinker(train_file_path=params['schema_train'], dev_file_path=params['schema_dev'], test_file_path=params['schema_test'])
        self.ranking_candidate = LogicalFormRetriever(dev_file_path=params['lf_dev'], test_file_path=params['lf_test'])
        self.retriever = FreebaseRetriever(freebase_addr=params['freebase_addr'], freebase_port=params['freebase_port'])

        if mode == 'qa':
            # models
            self.model_name = params.get('model_name', 't5-base')  # 'google/t5-v1_1-base'
            self.train_batch_size = 8
            self.eval_batch_size = 32
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
            if params['golden_schema'].lower() == 'true':
                self.max_source_length = 512
            elif 'lf' in self.prompt and 'schema' in self.prompt:
                self.max_source_length = 1000
            elif 'schema' in self.prompt and 'lf' not in self.prompt:
                self.max_source_length = 384
            else:
                self.max_source_length = 128
            self.max_target_length = 128
            self.model_eval(params['grailqa_generation'])

            # generation settings
            self.num_beams = params.get('num_beams', 10)
            self.logits_processor_list = LogitsProcessorList([])  # empty logits proccessor list
            if self.checker:
                self.logits_processor_list = LogitsProcessorList([SExpressionLogitsProcessor(self.tokenizer, 'grail_qa')])

    def model_eval(self, model_dir):
        if os.path.isfile(model_dir + '/' + self.prompt + '/pytorch_model.bin'):
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir + '/' + self.prompt)
            self.model.eval()
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)

    def encode(self, dataloader: GrailQAJsonLoader, output_path, task_prefix='translate English to Lisp: '):
        encodings_path = output_path + '/' + dataloader.get_dataset_split() + '_encodings'
        if output_path is not None and os.path.isfile(encodings_path):
            res = pickle_load(encodings_path)
            return res

        input_sequences = []
        output_sequences = []

        for idx in tqdm(range(0, dataloader.len)):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            s_expression = dataloader.get_s_expression_by_idx(idx)
            entities, relations, classes, rng_candidate_query = self.get_entity_schema_lf(dataloader, idx, question_id)
            input_seq = get_question_with_entity_and_schema(get_question_with_candidate_query(question, rng_candidate_query),
                                                            entities, relations, classes, entity_mid=True)
            input_sequences.append(input_seq)
            output_seq = s_expression
            output_sequences.append(output_seq)
        # end for each question

        print('encoding ', len(input_sequences))
        # encode the inputs
        encodings = self.tokenizer([task_prefix + sequence for sequence in input_sequences], padding='max_length', max_length=self.max_source_length, truncation=True)
        input_ids, attention_mask = encodings.input_ids, encodings.attention_mask
        # print example
        print('[Example]')
        for i in range(0, 5):
            print('[Input]' + input_sequences[i])
            print('[Output]' + output_sequences[i])
            print('[Input token len]' + str(len(self.tokenizer.tokenize(input_sequences[i]))))
            print('\n')

        res = {'input_ids': input_ids, 'attention_mask': attention_mask}

        # encode the targets
        if len(output_sequences):
            target_encoding = self.tokenizer(output_sequences, padding='max_length', max_length=self.max_target_length, truncation=True)
            labels = target_encoding.input_ids

            # replace padding token id's of the labels by -100
            labels = torch.tensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100
            res['labels'] = labels

        pickle_save(res, encodings_path)
        return res

    def get_entity_schema_lf(self, dataloader, idx, question_id):
        if dataloader.get_dataset_split() != 'test':
            golden_entities = dataloader.get_golden_entity_by_idx(idx)
            golden_relations = dataloader.get_golden_relation_by_idx(idx)
            golden_classes = dataloader.get_golden_class_by_idx(idx, True)
        candidate_classes, candidate_entities, candidate_relations, rng_candidate_query = self.get_lf_and_linked_schema(question_id)
        if dataloader.get_dataset_split() == 'test':
            return candidate_entities, candidate_relations, candidate_classes, rng_candidate_query
        if self.params['golden_entity'].lower() == 'true':
            entities = golden_entities
        else:
            entities = candidate_entities
        if self.params['golden_schema'].lower() == 'true':
            relations = golden_relations
            classes = golden_classes
        else:
            relations = candidate_relations
            classes = candidate_classes
        return entities, relations, classes, rng_candidate_query

    def get_lf_and_linked_schema(self, question_id):
        candidate_entities = self.grailqa_entity_linker.get_entities_by_question_id(question_id)
        entity_id_set = set([entity['id'] for entity in candidate_entities])
        candidate_relations = None
        candidate_classes = None
        rng_candidate_query = None

        if 'lf' in self.prompt:
            rng_candidate_query = self.ranking_candidate.get_logical_form_by_question_id(question_id)
            if rng_candidate_query is not None:
                for lf in rng_candidate_query:
                    entities, _ = get_entity_schema_in_lisp(lf)
                    for e in entities:
                        if e not in entity_id_set:
                            name = self.retriever.entity_name_by_mid(e)
                            if isinstance(name, list):
                                name = name[0]
                            candidate_entities.append({'id': e, 'friendly_name': name})
                            entity_id_set.add(e)

        if 'schema' in self.prompt:
            schema_beyond_lf = ('beyond' in self.prompt and rng_candidate_query is not None)
            top_k = 50 if schema_beyond_lf else 10
            candidate_relations = self.grailqa_schema_linker.get_relation_by_question_id(question_id, top_k=top_k)
            candidate_classes = self.grailqa_schema_linker.get_class_by_question_id(question_id, top_k=top_k)

            assert candidate_classes is not None and len(candidate_classes)
            assert candidate_relations is not None and len(candidate_relations)

            if schema_beyond_lf:
                new_relations = []
                new_classes = []
                for r in candidate_relations:
                    not_appear = True
                    for query in rng_candidate_query:
                        if is_schema_in_lisp(r, query):
                            not_appear = False
                            break
                    if not_appear:
                        new_relations.append(r)
                for c in candidate_classes:
                    not_appear = True
                    for query in rng_candidate_query:
                        if is_schema_in_lisp(c, query):
                            not_appear = False
                    if not_appear:
                        new_classes.append(c)
                candidate_relations = new_relations[:15]
                candidate_classes = new_classes[:15]
        return candidate_classes, candidate_entities, candidate_relations, rng_candidate_query

    def train(self, train_dataloader, dev_dataloader, output_dir):
        output_dir = output_dir + '/' + self.prompt
        assert output_dir is not None and os.path.isdir(output_dir)
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('Model already exists in {}'.format(output_dir))
            return

        train_encodings = self.encode(train_dataloader, output_dir.rstrip('/'))
        dev_encodings = self.encode(dev_dataloader, output_dir.rstrip('/'))

        train_dataset = HFDataset2(train_encodings)
        dev_dataset = HFDataset2(dev_encodings)

        # training settings
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        metric = load_metric('sacrebleu')

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]
            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        training_args = Seq2SeqTrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                                 # evaluation_strategy=IntervalStrategy.STEPS, save_strategy=IntervalStrategy.STEPS, eval_steps=500,
                                                 # eval_accumulation_steps=200,
                                                 evaluation_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.EPOCH,
                                                 per_device_train_batch_size=self.train_batch_size, per_device_eval_batch_size=self.eval_batch_size, num_train_epochs=10,
                                                 learning_rate=3e-5,
                                                 load_best_model_at_end=False)
        trainer = Seq2SeqTrainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
        best_run = trainer.train()

        trainer.save_model(output_dir)

    def solve(self, dataloader: GrailQAJsonLoader):
        logs = []
        logs_with_ans = {}
        time_str = time.strftime('_%Y_%m_%d_%H_%M_%S')

        metrics = Metrics()
        for idx in tqdm(range(0, dataloader.get_len())):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            golden_s_expression = dataloader.get_s_expression_by_idx(idx)
            golden_ans = dataloader.get_ans_arg_by_idx(idx)

            entities, relations, classes, rng_candidate_query = self.get_entity_schema_lf(dataloader, idx, question_id)
            model_input = get_question_with_entity_and_schema(get_question_with_candidate_query(question, rng_candidate_query),
                                                              entities, relations, classes, entity_mid=True)
            predicted, pred_score = self.conditional_generation(model_input)

            # metrics
            valid_query = ''
            rank = precision = recall = f1 = em = str_em = 0
            predicted_ans = []
            for pred in predicted:
                rank += 1
                try:
                    predicted_sparql = lisp_to_sparql(pred)
                    predicted_ans = self.retriever.query_var(predicted_sparql, 'x')
                    if len(predicted_ans) == 0 or len(predicted_ans[0]) == 0:
                        continue
                    valid_query = pred
                    precision = get_precision(golden_ans, predicted_ans)
                    recall = get_recall(golden_ans, predicted_ans)
                    f1 = get_f1_score_by_pr(precision, recall)
                    metrics.add_metric('P', precision)
                    metrics.add_metric('R', recall)
                    metrics.add_metric('F1', f1)
                    em = 1.0 if f1 == 1.0 else 0.0
                    metrics.add_metric('EM', em)
                    str_em = 1.0 if pred == golden_s_expression else 0.0
                    metrics.add_metric('str EM', str_em)
                    break
                except:
                    continue
            if valid_query == '':
                metrics.add_metric('str EM', 0.0)
                rank = -1
            metrics.count()

            # log this question
            avg_precision = metrics.get_metric('P')
            avg_recall = metrics.get_metric('R')
            avg_F1 = metrics.get_metric('F1')
            avg_em = metrics.get_metric('EM')
            avg_str_em = metrics.get_metric('str EM')
            log = {'serial': idx, 'qid': question_id, 'question': question, 'golden_s_expr': golden_s_expression, 'predicted_s_expr': valid_query,
                   'top_predictions': predicted, 'top_scores': pred_score, 'prediction_rank': rank,
                   'P': precision, 'R': recall, 'F1': f1, 'EM': em, 'str EM': str_em,
                   'avg_P': avg_precision, 'avg_R': avg_recall, 'avg_F1': avg_F1, 'avg_EM': avg_em, 'avg_str_EM': avg_str_em}
            logs.append(log)
            print('[' + str(idx) + ']', 'P:', avg_precision, 'R:', avg_recall, 'F1:', avg_F1, 'EM:', avg_em, 'str EM:', avg_str_em)

            with open('../logs/grailqa' + time_str + '.jsonl', 'a+') as f:
                f.write(json.dumps(log) + '\n')

            # back to ranking results
            predicted_ans, valid_query = self.backing_to_rank(predicted_ans, valid_query, rng_candidate_query)
            logs_with_ans[question_id] = {'logical_form': valid_query, 'rank': rank, 'answer': predicted_ans}
        # end for each question
        # save_log(logs, 'grailqa', time_str)
        save_log(logs_with_ans, 'grailqa_ans', time_str)

    def backing_to_rank(self, predicted_ans, predicted_query, rng_candidate_query):
        if self.back_to_rank is True and (predicted_query == '' or len(predicted_ans) == 0 or predicted_ans[0] == '0'):
            if rng_candidate_query is not None and len(rng_candidate_query):
                for lf in rng_candidate_query:
                    try:
                        sparql = lisp_to_sparql(lf)
                        ans = self.retriever.query_var(sparql, 'x')
                        if len(ans) and ans[0] != '0':
                            predicted_query = lf
                            predicted_ans = ans
                            break
                    except:
                        continue
        return predicted_ans, predicted_query

    def solve_test(self, dataloader: GrailQAJsonLoader, query_norm=False):
        logs_with_ans = dict()
        time_str = time.strftime('_%Y_%m_%d_%H_%M_%S')
        split = dataloader.get_dataset_split()

        for idx in tqdm(range(dataloader.get_len())):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)

            entities, relations, classes, rng_candidate_query = self.get_entity_schema_lf(dataloader, idx, question_id)
            model_input = get_question_with_entity_and_schema(get_question_with_candidate_query(question, rng_candidate_query), entities, relations, classes, entity_mid=True)
            predicted, pred_score = self.conditional_generation(model_input)

            # metrics
            rank = 0
            valid_query = ''
            predicted_ans = []
            for pred in predicted:
                rank += 1
                try:
                    predicted_sparql = lisp_to_sparql(pred)
                    predicted_ans = self.retriever.query_var(predicted_sparql, 'x')
                    if len(predicted_ans) == 0 or len(predicted_ans[0]) == 0:  # empty answer
                        continue
                    valid_query = pred  # not empty answer
                    break
                except:
                    continue
            if valid_query == '':
                rank = -1

            # log this question
            predicted_ans, valid_query = self.backing_to_rank(predicted_ans, valid_query, rng_candidate_query)
            logs_with_ans[question_id] = {'logical_form': valid_query, 'answer': predicted_ans}
            if split != 'test':
                logs_with_ans[question_id]['rank'] = rank
            print('[' + str(idx) + ']', question, valid_query, predicted_ans)

        # end for each question
        save_log(logs_with_ans, 'grailqa_' + dataloader.get_dataset_split(), time_str)

    def conditional_generation(self, question: str, task_prefix='translate English to Lisp: '):
        input_ids = self.tokenizer(task_prefix + question, return_tensors='pt', max_length=self.max_source_length, truncation=True, padding="max_length").input_ids
        outputs = self.model.generate(input_ids.to(self.device), max_length=self.max_target_length, num_beams=self.num_beams, num_return_sequences=self.num_beams,
                                      logits_processor=self.logits_processor_list, output_scores=True, return_dict_in_generate=True)
        pred_expr = []
        pred_score = []
        outputs_seq = outputs['sequences']
        for i in range(0, len(outputs_seq)):
            output = self.tokenizer.decode(outputs_seq[i], skip_special_tokens=True)
            if '^^' not in output:
                output = output.replace('http://www.w3.org/2001/XMLSchema#', '^^http://www.w3.org/2001/XMLSchema#')
            pred_expr.append(output)
            pred_score.append(float(outputs['sequences_scores'][i]))
        return pred_expr, pred_score


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=input_path, required=False)
    parser.add_argument('--output_path', type=str, default=output_path, required=False)
    parser.add_argument('--freebase_address', type=str, default=freebase_addr, required=False)
    parser.add_argument('--freebase_port', type=str, default=freebase_port, required=False)
    parser.add_argument('--prompt', type=str, default='lf_schema', required=False, choices=['lf_schema', 'lf', 'schema', 'none'])
    parser.add_argument('--checker', type=str, default='True', required=False)
    parser.add_argument('--run_valid', type=str, default='True', required=False)
    parser.add_argument('--run_test', type=str, default='False', required=False)
    parser.add_argument('--golden_entity', type=str, default='False', required=False)
    parser.add_argument('--golden_schema', type=str, default='False', required=False)
    parser.add_argument('--verbose', type=str, default='False', required=False)
    parser.add_argument('--model_name', type=str, default='t5-base', required=False)
    args = parser.parse_args()

    input_path = args.input_path.rstrip('/') + '/'
    output_path = args.output_path.rstrip('/') + '/'
    params = dict()
    params['input_path'] = input_path
    params['output_path'] = output_path
    params['freebase_addr'] = args.freebase_address
    params['freebase_port'] = args.freebase_port
    params['grailqa_train'] = input_path + grailqa_train_path.lstrip('./')
    params['grailqa_dev'] = input_path + grailqa_dev_path.lstrip('./')
    params['grailqa_test'] = input_path + grailqa_test_path.lstrip('./')
    params['freebase_types'] = input_path + grailqa_fb_types_path.lstrip('./')
    params['grailqa_generation'] = output_path + grailqa_generation_model_path.lstrip('./')
    params['grailqa_dev_el'] = input_path + grailqa_tiara_dev_el_path.lstrip('./')
    params['grailqa_test_el'] = input_path + grailqa_tiara_test_el_path.lstrip('./')
    params['schema_train'] = input_path + grailqa_schema_train_path.lstrip('./')
    params['schema_dev'] = input_path + grailqa_schema_dev_path.lstrip('./')
    params['schema_test'] = input_path + grailqa_schema_test_path.lstrip('./')
    params['lf_train'] = input_path + grailqa_rng_ranking_train_path.lstrip('./')
    params['lf_dev'] = input_path + grailqa_tiara_ranking_dev_path.lstrip('./')
    params['lf_test'] = input_path + grailqa_tiara_ranking_test_path.lstrip('./')
    params['model_name'] = args.model_name
    params['golden_entity'] = args.golden_entity
    params['golden_schema'] = args.golden_schema
    params['prompt'] = args.prompt
    params['checker'] = args.checker
    # arg end

    grail_train_data = GrailQAJsonLoader(params['grailqa_train'])
    grail_dev_data = GrailQAJsonLoader(params['grailqa_dev'])
    grail_test_data = GrailQAJsonLoader(params['grailqa_test'])

    grail_qa_algorithm = GrailQAGeneration(params)
    grail_qa_algorithm.train(grail_train_data, grail_dev_data, params['grailqa_generation'])
    verbose = string_to_bool(args.verbose)
    if args.run_valid.lower() == 'true':
        if not verbose:
            grail_qa_algorithm.solve_test(grail_dev_data)
        else:
            grail_qa_algorithm.solve(grail_dev_data)
    if args.run_test.lower() == 'true':
        if not verbose:
            grail_qa_algorithm.solve_test(grail_test_data)
        else:
            grail_qa_algorithm.solve(grail_test_data)
