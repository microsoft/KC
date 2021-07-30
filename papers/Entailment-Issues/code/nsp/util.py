import codecs
from collections import defaultdict
import numpy as np
import statistics
import logging
from transformers.tokenization_bert import BertTokenizer
import os, json
import random
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, inst_id, input_ids, input_mask, segment_ids, label_id):
        self.inst_id = inst_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    premise_2_tokenzed = {}
    hypothesis_2_tokenzed = {}
    list_2_tokenizedID = {}
    example_strs = []
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * (len(tokens_b) + 1)
        tokens = tokens_A + tokens_B
        segment_ids = segment_ids_A + segment_ids_B

        example_strs.append(" ".join(tokens_A) + " " + " ".join(tokens_B))

        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
            list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B
        input_ids = input_ids_A + input_ids_B

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 11:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(inst_id = ex_index,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features, example_strs


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def load_label_fn(label_fn):
    with open(label_fn, mode="r", encoding="utf-8") as fp:
        first = True
        des = ""
        for line in fp:
            if first:
                print(line)
                classes = json.loads(line)
                first = False
            else:
                des += line
        type2hypothesis = json.loads(des)    
    return classes, type2hypothesis

def random_text_fn(text_fn, rand_fn):
    lines = []
    with open(text_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            text, label = line.strip("\n").split("\t")
            tmp = text.split()
            random.shuffle(tmp)
            text = ' '.join(tmp)
            lines.append(text + "\t" + label + "\n")
    with open(rand_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(lines)

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()       
                
class NSPData():
    def __init__(self, label_fn, input_fn=None, reverse=False):
        self.input_fn = input_fn
        self.classes, self.type2hypothesis = load_label_fn(label_fn)
        self.hypothesis2type = {}
        self.reverse = reverse
        for cname, hypo_list in self.type2hypothesis.items():
            for hypo in hypo_list:
                self.hypothesis2type[hypo] = cname
        
    def get_examples(self, input_fn=None, label_single=True):
        if input_fn is None:
            input_fn = self.input_fn
        gold_label_list = []
        line_co = 0
        exam_co = 0
        examples = []
        with open(input_fn, mode="r", encoding="utf-8") as fp:
            for row in fp:
                line = row.strip("\n").split("\t")
                if label_single:
                    assert len(line) == 2
                    gold_label_list.append(line[1])
                else:
                    gold_label_list.append(line[1:])
                for cname, hypo_list in self.type2hypothesis.items():
                    for hypo in hypo_list:
                        guid = str(exam_co)
                        if self.reverse:
                            text_a = hypo
                            text_b = line[0]
                        else:
                            text_a = line[0]
                            text_b = hypo
                        if (label_single and cname == gold_label_list[-1]):
                            label = 'isNext'
                        elif (not label_single) and (cname in gold_label_list[-1]):
                            label = 'isNext'
                        else:
                            label = 'Random'
                        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        exam_co += 1
                line_co += 1
                if line_co % 10000 == 0:
                    print('loading data size:', line_co)
        hypo_type_list = []
        for cname, hypo_list in self.type2hypothesis.items():
            for hypo in hypo_list:
                hypo_type_list.append((hypo, cname))
        return examples, gold_label_list, hypo_type_list
    
    def get_labels(self):
        return ['isNext', 'Random']