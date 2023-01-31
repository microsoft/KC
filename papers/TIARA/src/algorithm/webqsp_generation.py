# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')

import os
from retriever.ranking_candidate import LogicalFormRetriever
from utils.s_expr_util import execute_s_expr
from utils.question_pattern import get_question_with_entity_and_schema, get_question_with_candidate_query

from tqdm import tqdm
from utils.string_util import string_to_bool, get_entity_schema_in_lisp
from utils.webqsp_generation_data import WebQSPGenerationData
import json
import argparse
import time
from datasets import load_metric
import torch
from dataloader.webqsp_json_loader import WebQSPJsonLoader
from utils.file_util import pickle_load, pickle_save
from utils.hugging_face_dataset import HFDataset2
from utils.metrics import Metrics
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, IntervalStrategy, Seq2SeqTrainer, LogitsProcessorList
from retriever.entity_linker.webqsp_entity_linker import WebQSPEntityLinker
from retriever.freebase_retriever import FreebaseRetriever
from retriever.schema_linker.schema_file_reader import SchemaLinker
from utils.config import freebase_addr, freebase_port, webqsp_test_path, webqsp_schema_test_path, webqsp_rng_ranking_train_path, webqsp_rng_ranking_test_path, \
    webqsp_generation_model_path, webqsp_pdev_path, \
    webqsp_ptrain_path, webqsp_schema_pdev_path, webqsp_schema_ptrain_path, webqsp_rng_oracle_entity_ranking_test_path


class WebQSPGeneration:
    def __init__(self, params):
        self.params = params
        self.prompt = params.get('prompt', 'lf_relation')

        self.retriever = FreebaseRetriever(freebase_addr=params['freebase_addr'], freebase_port=params['freebase_port'])
        self.webqsp_entity_linker = WebQSPEntityLinker(self.retriever)
        self.webqsp_schema_linker = SchemaLinker(train_file_path=params['schema_train'], dev_file_path=params['schema_dev'], test_file_path=params['schema_test'],
                                                 file_format='json')
        self.ranking_candidate = LogicalFormRetriever(train_file_path=params['lf_train'], dev_file_path=None, test_file_path=params['lf_test'])
        self.generation_data = WebQSPGenerationData()

        # models
        self.model_name = params.get('model_name', 't5-base')  # 'google/t5-v_1_1-base'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        self.warmup_ratio = 0
        if 'lf' in self.prompt:
            if 'schema' in self.prompt:
                self.max_source_length = 1000
            elif 'relation' in self.prompt:
                self.max_source_length = 768
            else:
                self.max_source_length = 512
        else:
            if 'schema' in self.prompt:
                self.max_source_length = 512
            elif 'relation' in self.prompt:
                self.max_source_length = 256
            else:
                self.max_source_length = 80
        self.max_target_length = 128
        self.model_eval(params['webqsp_generation'])
        self.num_epoch = 20
        self.golden_entity = params.get('golden_entity')

        if 'lf' not in self.prompt:
            self.back_to_rank = False
        else:
            self.back_to_rank = params.get('back_to_rank', True)

        # generation settings
        self.num_beams = params.get('num_beams', 10)
        self.logits_processor_list = LogitsProcessorList()
        print('golden_entity: {}, prompt: {}, back_to_rank: {}, max_source_len: {}'.format(self.golden_entity, self.prompt, self.back_to_rank,
                                                                                           self.max_source_length))
        self.task_prefix = 'translate English to Lisp: '

    def model_eval(self, model_dir):
        if os.path.isfile(model_dir + '/' + self.prompt + '/pytorch_model.bin'):
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir + '/' + self.prompt)
            self.model.eval()
            # self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)

    def get_lf_and_linked_schema(self, question_id, oracle_entity):
        candidate_entities = self.webqsp_entity_linker.get_entities_by_question_id(question_id, only_id=False, oracle=oracle_entity)
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
                        if e in entity_id_set:
                            continue
                        name = self.retriever.entity_name_by_mid(e)
                        if isinstance(name, list):
                            name = name[0]
                        candidate_entities.append({'id': e, 'friendly_name': name})
                        entity_id_set.add(e)
        if 'schema' in self.prompt:
            candidate_relations = self.webqsp_schema_linker.get_relation_by_question_id(question_id, top_k=10)
            candidate_classes = self.webqsp_schema_linker.get_class_by_question_id(question_id, top_k=10)
        elif 'relation' in self.prompt:
            candidate_relations = self.webqsp_schema_linker.get_relation_by_question_id(question_id, top_k=10)
        return candidate_classes, candidate_entities, candidate_relations, rng_candidate_query

    def encode(self, dataloader: WebQSPJsonLoader, output_path):
        encodings_path = output_path.rstrip('/') + '/' + dataloader.get_dataset_split() + '_encodings'
        if output_path is not None and os.path.isdir(output_path):
            if os.path.isfile(encodings_path):  # file exists
                res = pickle_load(encodings_path)
                return res

        input_sequences = []
        output_sequences = []

        for idx in tqdm(range(dataloader.len)):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            generation_target = self.generation_data.get_generation_target_by_question_id(question_id)

            if generation_target is None or generation_target == 'null':
                print('[WARN] s-expression is null for question {}'.format(question_id))
                continue
            # input
            if dataloader.get_dataset_split() == 'train' or self.golden_entity is True:
                oracle = True
            else:
                oracle = False

            classes, entities, relations, rng_candidate_query = self.get_lf_and_linked_schema(question_id, oracle_entity=oracle)
            input_seq = get_question_with_entity_and_schema(get_question_with_candidate_query(question, rng_candidate_query),
                                                            entities, relations, classes, entity_mid=True)
            input_sequences.append(input_seq)
            # output
            output_sequences.append(generation_target)
        # end for each question

        print('encoding ', len(input_sequences))
        # encode the inputs
        encodings = self.tokenizer([self.task_prefix + sequence for sequence in input_sequences], padding='max_length', max_length=self.max_source_length, truncation=True)
        input_ids, attention_mask = encodings.input_ids, encodings.attention_mask
        # print example
        print('[Example]')
        for i in range(0, 5):
            print('[Input]' + input_sequences[i])
            print('[Output]' + output_sequences[i])
            print('[Input token len]' + str(len(self.tokenizer.tokenize(input_sequences[i]))))
            print()

        max_token_len = 0
        avg_token_len = 0
        for input_seq in input_sequences:
            l = len(self.tokenizer.tokenize(input_seq))
            max_token_len = max(max_token_len, l)
            avg_token_len += l
        avg_token_len /= len(input_sequences)
        print('[max token len]', max_token_len, '[avg token len]', avg_token_len)

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

        training_args = Seq2SeqTrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                                 # evaluation_strategy=IntervalStrategy.STEPS, save_strategy=IntervalStrategy.STEPS, eval_steps=500,
                                                 # eval_accumulation_steps=200,
                                                 evaluation_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.EPOCH,
                                                 per_device_train_batch_size=2, per_device_eval_batch_size=8, num_train_epochs=self.num_epoch, learning_rate=3e-5,
                                                 load_best_model_at_end=False, warmup_ratio=self.warmup_ratio)
        trainer = Seq2SeqTrainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
        best_run = trainer.train()

        trainer.save_model(output_dir)

    def solve(self, dataloader: WebQSPJsonLoader):
        logs = []
        time_str = time.strftime('_%Y_%m_%d_%H_%M_%S')

        metrics = Metrics()
        for idx in tqdm(range(dataloader.len)):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            generation_target = self.generation_data.get_generation_target_by_question_id(question_id)

            classes, entities, relations, rng_candidate_query = self.get_lf_and_linked_schema(question_id, oracle_entity=self.golden_entity)
            model_input = get_question_with_entity_and_schema(get_question_with_candidate_query(question, rng_candidate_query),
                                                              entities, relations, classes, entity_mid=True)
            predicted, pred_score = self.conditional_generation(model_input)

            # metrics
            valid_query = ''
            rank = 0
            predicted_ans = []
            for pred in predicted:
                rank += 1
                try:
                    pred, predicted_ans = execute_s_expr(pred)
                    if len(predicted_ans) == 0 or len(predicted_ans[0]) == 0:
                        continue
                    valid_query = pred
                    if len(predicted_ans) > 0:  # there's answer
                        break
                except:
                    continue
            if valid_query == '' or valid_query == 'null':
                rank = -1

            predicted_ans, valid_query = self.backing_to_rank(predicted_ans, valid_query, rng_candidate_query)
            precision, recall, f1, em = self.eval(dataloader, predicted_ans, metrics, idx)

            # log this question
            avg_precision = metrics.get_metric('P')
            avg_recall = metrics.get_metric('R')
            avg_F1 = metrics.get_metric('F1')
            avg_em = metrics.get_metric('EM')
            log = {'serial': idx, 'qid': question_id, 'question': question, 'golden_s_expr': generation_target, 'predicted_s_expr': valid_query,
                   'top_predictions': predicted, 'prediction_rank': rank, 'P': precision, 'R': recall, 'F1': f1, 'EM': em,
                   'avg_P': avg_precision, 'avg_R': avg_recall, 'avg_F1': avg_F1, 'avg_EM': avg_em,
                   'answer': predicted_ans}
            logs.append(log)
            print('[' + str(idx) + ']', 'P:', avg_precision, 'R:', avg_recall, 'F1:', avg_F1, 'EM:', avg_em)

            with open('../logs/webqsp' + time_str + '.jsonl', 'a+') as f:
                f.write(json.dumps(log) + '\n')
        # end for each question
        # save_log(logs, 'webqsp', time_str)  # save to jsonfile

    def eval(self, dataloader: WebQSPJsonLoader, prediction, metrics, ques_idx):
        skip = True
        for pidx in range(0, len(dataloader.data[ques_idx]['Parses'])):  # for each parse
            np = dataloader.data[ques_idx]['Parses'][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False  # if all parses are not good and complete, skip this question
        if len(dataloader.data[ques_idx]['Parses']) == 0 or skip:
            return 'n/a', 'n/a', 'n/a', 'n/a'

        metrics.count()
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        for pidx in range(0, len(dataloader.data[ques_idx]['Parses'])):  # for each parse
            golden = dataloader.data[ques_idx]['Parses'][pidx]['Answers']
            p, r, f1 = self.get_p_r_f1(golden, prediction)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = p
                best_recall = r

        best_em = 1.0 if best_f1 == 1.0 else 0.0
        metrics.add_metric('P', best_precision)
        metrics.add_metric('R', best_recall)
        metrics.add_metric('F1', best_f1)
        metrics.add_metric('EM', best_em)
        return best_precision, best_recall, best_f1, best_em

    def get_p_r_f1(self, golden, pred):
        if len(golden) == 0:
            if len(pred) == 0:
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 1.0, 0.0
        elif len(pred) == 0:
            return 1.0, 0.0, 0.0
        else:
            golden = [x["AnswerArgument"] for x in golden]

            tp = 1e-40  # numerical trick
            fp = 0.0
            fn = 0.0

            for g in golden:
                if g in pred:
                    tp += 1
                else:
                    fn += 1
            for p in pred:
                if p not in golden:
                    fp += 1

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1

    def conditional_generation(self, question: str):
        input_ids = self.tokenizer(self.task_prefix + question, return_tensors='pt', max_length=self.max_source_length, truncation=True, padding="max_length").input_ids
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

    def backing_to_rank(self, predicted_ans, predicted_query, rng_candidate_query):
        if self.back_to_rank is True and (predicted_query == '' or predicted_query == 'null' or len(predicted_ans) == 0):
            if rng_candidate_query is not None and len(rng_candidate_query):
                for lf in rng_candidate_query:
                    try:
                        lf, ans = execute_s_expr(lf)
                        if len(ans) and ans[0] != '0':
                            predicted_query = lf
                            predicted_ans = ans
                            break
                    except:
                        continue
        return predicted_ans, predicted_query


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--freebase_address', type=str, default=freebase_addr, required=False)
    parser.add_argument('--freebase_port', type=str, default=freebase_port, required=False)
    parser.add_argument('--prompt', type=str, default='lf_relation', required=False)
    parser.add_argument('--run_valid', type=str, default='False', required=False)
    parser.add_argument('--run_test', type=str, default='True', required=False)
    parser.add_argument('--model_dir', type=str, default=webqsp_generation_model_path, required=False)
    parser.add_argument('--model_name', type=str, default='t5-base', required=False)
    parser.add_argument('--golden_entity', type=str, default='False', required=False)
    args = parser.parse_args()

    params = dict()
    params['webqsp_generation'] = webqsp_generation_model_path
    params['freebase_addr'] = args.freebase_address
    params['freebase_port'] = args.freebase_port
    params['prompt'] = args.prompt
    params['webqsp_train'] = webqsp_ptrain_path
    params['webqsp_dev'] = webqsp_pdev_path
    params['webqsp_test'] = webqsp_test_path
    params['schema_train'] = webqsp_schema_ptrain_path
    params['schema_dev'] = webqsp_schema_pdev_path
    params['schema_test'] = webqsp_schema_test_path
    params['model_name'] = args.model_name
    params['lf_train'] = webqsp_rng_ranking_train_path
    golden_entity = string_to_bool(args.golden_entity)
    if golden_entity:  # oracle entity
        params['lf_test'] = webqsp_rng_oracle_entity_ranking_test_path
        params['golden_entity'] = True
    else:  # entity linking
        params['lf_test'] = webqsp_rng_ranking_test_path
        params['golden_entity'] = False

    webqsp_train_data = WebQSPJsonLoader(params['webqsp_train'])
    webqsp_dev_data = WebQSPJsonLoader(params['webqsp_dev'])
    webqsp_test_data = WebQSPJsonLoader(params['webqsp_test'])

    webqsp_qa_algorithm = WebQSPGeneration(params)
    webqsp_qa_algorithm.train(webqsp_train_data, webqsp_dev_data, params['webqsp_generation'])
    if args.run_valid.lower() == 'true':
        webqsp_qa_algorithm.solve(webqsp_dev_data)
    if args.run_test.lower() == 'true':
        webqsp_qa_algorithm.solve(webqsp_test_data)
