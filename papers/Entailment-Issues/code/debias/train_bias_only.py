'''
The code is adapted from https://github.com/chrisc36/debias/blob/master/debias/preprocessing/build_mnli_bias_only.py

License: Apache License 2.0
'''

import argparse
import logging
import pickle
from os import mkdir
from os.path import exists, join

from sklearn.linear_model import LogisticRegression
from collections import namedtuple
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Iterable, TypeVar

import nltk
import numpy as np
import regex
from tqdm import tqdm
from utils import flatten_list
from load_data import *

_double_quote_re = regex.compile(u"\"|``|''")


def convert_to_spans(raw_text: str, text: List[str]) -> np.ndarray:
    cur_idx = 0
    all_spans = np.zeros((len(text), 2), dtype=np.int32)
    for i, token in enumerate(text):
        if _double_quote_re.match(token):
            span = _double_quote_re.search(raw_text[cur_idx:])
            tmp = cur_idx + span.start()
            l = span.end() - span.start()
        else:
            tmp = raw_text.find(token, cur_idx)
            l = len(token)

        if tmp < cur_idx:
            raise ValueError(token)
        cur_idx = tmp
        all_spans[i] = (cur_idx, cur_idx + l)
        cur_idx += l
    return all_spans

class NltkAndPunctTokenizer():
    """Tokenize ntlk, but additionally split on most punctuations symbols"""
    def __init__(self, split_dash=True, split_single_quote=False, split_period=False, split_comma=False):
        self.split_dash = split_dash
        self.split_single_quote = split_single_quote
        self.split_period = split_period
        self.split_comma = split_comma

        # Unix character classes to split on
        resplit = r"\p{Pd}\p{Po}\p{Pe}\p{S}\p{Pc}"

        # A list of optional exceptions, will we trust nltk to split them correctly
        # unless otherwise specified by the ini arguments
        dont_split = ""
        if not split_dash:
            dont_split += "\-"
        if not split_single_quote:
            dont_split += "'"
        if not split_period:
            dont_split += "\."
        if not split_comma:
            dont_split += ","

        resplit = "([" + resplit + "]|'')"
        if len(dont_split) > 0:
            split_regex = r"(?![" + dont_split + "])" + resplit
        else:
            split_regex = resplit

        self.split_regex = regex.compile(split_regex)
        try:
            self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
        except LookupError:
            logging.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
            self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')

        self.word_tokenizer = nltk.TreebankWordTokenizer()

    def retokenize(self, x):
        if _double_quote_re.match(x):
            return (x, )
        return (x.strip() for x in self.split_regex.split(x) if len(x) > 0)

    def tokenize(self, text: str) -> List[str]:
        out = []
        for s in self.sent_tokenzier.tokenize(text):
            out += flatten_list(self.retokenize(w) for w in self.word_tokenizer.tokenize(s))
        return out

    def tokenize_with_inverse(self, paragraph: str):
        text = self.tokenize(paragraph)
        inv = convert_to_spans(paragraph, text)
        return text, inv
    

def tokenize_examples(data, tokenizer):
    out = []
    for example in data:
        out.append(TextPairExample(
            example.id,
            tokenizer.tokenize(example.premise),
            tokenizer.tokenize(example.hypothesis),
            example.label
          ))
    return out    


STOP_WORDS = frozenset([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
  'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
  'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
  'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
  've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
  'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
   "many", "how", "de"
])

def is_subseq(needle, haystack):
    l = len(needle)
    if l > len(haystack):
        return False
    else:
        return any(haystack[i:i+l] == needle for i in range(len(haystack)-l + 1))


def build_mnli_bias_only(args):
    """Builds our bias-only MNLI model and saves its predictions
    :param out_dir: Directory to save the predictions
    :param cache_examples: Cache examples to this file
    :param w2v_cache: Cache w2v features to this file
    """
    tok = NltkAndPunctTokenizer()

    # Load the data we want to use
    if args.cache_examples and exists(args.cache_examples):
        with open(args.cache_examples, "rb") as f:
            dataset_to_examples = pickle.load(f)
    else:
        dataset_to_examples = {}
        dataset_to_examples["hans"] = tokenize_examples(load_hans(args), tok)
        dataset_to_examples["train"] = tokenize_examples(load_mnli(args, True), tok)
        dataset_to_examples["dev"] = tokenize_examples(load_mnli(args, False), tok)
        if args.cache_examples:
            with open(args.cache_examples, "wb") as f:
                pickle.dump(dataset_to_examples, f)


    # Load the pre-normalized word vectors to use when building features
    if args.w2v_cache and exists(args.w2v_cache):
        with open(args.w2v_cache, "rb") as f:
            w2v = pickle.load(f)
    else:
        logging.info("Loading word vectors")
        voc = set()
        for v in dataset_to_examples.values():
            for ex in v:
                voc.update(ex.hypothesis)
                voc.update(ex.premise)
        words, vecs = load_word_vectors(args.w2v_fn, voc)
        w2v = {w: v/np.linalg.norm(v) for w, v in zip(words, vecs)}
        if args.w2v_cache:
            with open(args.w2v_cache, "wb") as f:
                pickle.dump(w2v, f)

    # Build the features, store as a pandas dataset
    dataset_to_features = {}
    for name, examples in dataset_to_examples.items():
        features = []
        for example in examples:
            h = [x.lower() for x in example.hypothesis]
            p = [x.lower() for x in example.premise]
#             print(h)
#             print(p)
            p_words = set(p)
            n_words_in_p = sum(x in p_words for x in h)
            fe = {
                "h-is-subseq": is_subseq(h, p),
                "all-in-p": n_words_in_p == len(h),
                "percent-in-p": n_words_in_p / len(h),
                "log-len-diff": np.log(max(len(p) - len(h), 1)),
                "label": example.label
             }

            h_vecs = [w2v[w] for w in example.hypothesis if w in w2v]
            p_vecs = [w2v[w] for w in example.premise if w in w2v]
            if len(h_vecs) > 0 and len(p_vecs) > 0:
                h_vecs = np.stack(h_vecs, 0)
                p_vecs = np.stack(p_vecs, 0)
                # [h_size, p_size]
                similarities = np.matmul(h_vecs, p_vecs.T)
                # [h_size]
                similarities = np.max(similarities, 1)
                similarities.sort()
                fe["average-sim"] = similarities.sum() / len(h)
                fe["min-similarity"] = similarities[0]
                if len(similarities) > 1:
                    fe["min2-similarity"] = similarities[1]

            features.append(fe)
        dataset_to_features[name] = pd.DataFrame(features)
        dataset_to_features[name].fillna(0.0, inplace=True)

    # Train the model
    print("Fitting...")
    train_df = dataset_to_features["train"]
    feature_cols = [x for x in train_df.columns if x != "label"]

    # class_weight='balanced' will weight the entailemnt/non-entailment examples equally
    # C=100 means no regularization
    lr = LogisticRegression(multi_class="auto", solver="liblinear",
                          class_weight='balanced', C=100)
    lr.fit(train_df[feature_cols].values, train_df.label.values)

    # Save the model predictions
    if not exists(args.out_dir):
        mkdir(args.out_dir)

    for name, ds in dataset_to_features.items():
        print("Predicting for %s" % name)
        examples = dataset_to_examples[name]
        pred = lr.predict_log_proba(ds[feature_cols].values).astype(np.float32)
        y = ds.label.values

        bias = {}
        for i in range(len(pred)):
            if examples[i].id in bias:
                raise RuntimeError("non-unique IDs?")
            bias[examples[i].id] = pred[i]

        acc = np.mean(y == np.argmax(pred, 1))
        print("%s two-class accuracy: %.4f (size=%d)" % (name, acc, len(examples)))

        with open(join(args.out_dir, "%s.pkl" % name), "wb") as f:
            pickle.dump(bias, f)


def main():
    parser = argparse.ArgumentParser("Train our MNLI bias-only model")
    parser.add_argument("--input_dir", default=None, help="mnli data dir")
    parser.add_argument("--hans_dir", default=None, help="hans data dir")
    parser.add_argument("--out_dir", default=None, help="output result dir")
    parser.add_argument("--w2v_fn", default=None, help="word2vec file: crawl-300d-2M.vec")
    parser.add_argument("--cache_examples", default=None, help="file to cache mnli examples")
    parser.add_argument("--w2v_cache", default=None, help="path to cache w2v")
    args = parser.parse_args()

    build_mnli_bias_only(args)


if __name__ == "__main__":
    main()
