import os

import logging
import pickle
import json

from collections import namedtuple
from os import mkdir
from os.path import join, exists
from typing import List, Dict, Iterable, Tuple, Optional

import numpy as np
from tqdm import tqdm

import utils

# change the MNLI dataset fromat (3 way classification) to the two-way classification
NLI_LABELS = ["entailment", "not_entailment"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}

# change the MNLI dataset format; only contain 4 columns: id, premise, hypothesis, label
TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

def load_hans_subsets(args):
    src = join(args.hans_dir, "heuristics_evaluation_set.txt")
    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(args, filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets

# download hans https://raw.githubusercontent.com/hansanon/hans/master/heuristics_evaluation_set.txt
def load_hans(args, n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(args.hans_dir, "heuristics_evaluation_set.txt")

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = NLI_LABEL_MAP["not_entailment"]
        elif label == "entailment":
            label = NLI_LABEL_MAP["entailment"]
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out


def load_mnli(args, is_train, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filename = join(args.input_dir, "train.tsv")
    else:
        if custom_path is None:
            filename = join(args.input_dir, "dev.tsv") # processed dev_matched file
        else:
            filename = join(args.input_dir, custom_path)

    logging.info("Loading mnli " + ("train" if is_train else "dev"))
    with open(filename) as f:
        f.readline() # the first line corresponding the file head
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)
        with open(join(args.input_dir, "sample_train.tsv"), mode="w", encoding="utf-8") as fp:
            fp.writelines(lines)
        print("write to sample_train.tsv done .")

    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample(line[0], line[1], line[2], NLI_LABEL_MAP[line[-1].rstrip()]))
    return out


def load_bias(custom_path=None) -> Dict[str, np.ndarray]:
    """Load dictionary of example_id->bias where bias is a length 2 array
    of log-probabilities, this file is produced by train_bias_only.py"""

    if custom_path is not None:  # file contains probs
        print("load {} bias".format(custom_path))
        if "custom_path".endswith(".json"):
            with open(custom_path, "r") as bias_file:
                bias = json.load(bias_file)
        else:
            bias = utils.load_pickle(custom_path)
        for k, v in bias.items():
            bias[k] = np.array(v)
        return bias
    else:
        print("load no bias")
        return None
    

def load_word_vectors(vec_path: str, vocab: Optional[Iterable[str]]=None, n_words_to_scan=None):
    return load_word_vector_file(vec_path, vocab, n_words_to_scan)


def load_word_vector_file(vec_path: str, vocab: Optional[Iterable[str]]=None,
                          n_words_to_scan=None):
    if vocab is not None:
        vocab = set(vocab)
    if vec_path.endswith(".pkl"):
        with open(vec_path, "rb") as f:
            return pickle.load(f)

    # some of the large vec files produce utf-8 errors for some words, just skip them
    elif vec_path.endswith(".txt.gz"):
        handle = lambda x: gzip.open(x, 'r', encoding='utf-8', errors='ignore')
    else:
        handle = lambda x: open(x, 'r', encoding='utf-8', errors='ignore')

    if n_words_to_scan is None:
        if vocab is None:
            logging.info("Loading word vectors from %s..." % vec_path)
        else:
            logging.info("Loading word vectors from %s for voc size %d..." % (vec_path, len(vocab)))
    else:
        if vocab is None:
            logging.info("Loading up to %d word vectors from %s..." % (n_words_to_scan, vec_path))
        else:
            logging.info("Loading up to %d word vectors from %s for voc size %d..." % (n_words_to_scan, vec_path, len(vocab)))
    words = []
    vecs = []
    pbar = tqdm(desc="word-vec")
    with handle(vec_path) as fh:
        for i, line in enumerate(fh):
            pbar.update(1)
            if n_words_to_scan is not None and i >= n_words_to_scan:
                break
            word_ix = line.find(" ")
            if i == 0 and " " not in line[word_ix+1:]:
                # assume a header row, such as found in the fasttext word vectors
                print(line)
                continue
            word = line[:word_ix]
            if (vocab is None) or (word in vocab):
                words.append(word)
                vecs.append(np.fromstring(line[word_ix+1:], sep=" ", dtype=np.float32))
                if vecs[-1].shape[0] != 300:
                    print(vecs[-1].shape)
                    print("error")
    pbar.close()
    return words, vecs