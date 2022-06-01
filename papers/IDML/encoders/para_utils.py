'''
The code is adapted from https://github.com/jwieting/paraphrastic-representations-at-scale/blob/main/utils.py
'''

import torch
import random

unk_string = "UUUNKKK"

def max_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.max(x[i][0:lengths[i]], 0)[0]
    return out

def mean_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.mean(x[i][0:lengths[i]], 0)
    return out

def lookup(words, w, zero_unk):
    if w in words:
        return words[w]
    else:
        if zero_unk:
            return None
        else:
            return words[unk_string]

class Example(object):
    def __init__(self, sentence, lower_case):
        self.sentence = sentence.strip()
        if lower_case:
            self.sentence = self.sentence.lower()
        self.embeddings = []

    def populate_ngrams(self, sentence, words, zero_unk, n):
        sentence = " " + sentence.strip() + " "
        embeddings = []

        for j in range(len(sentence)):
            idx = j
            gr = ""
            while idx < j + n and idx < len(sentence):
                gr += sentence[idx]
                idx += 1
            if not len(gr) == n:
                continue
            wd = lookup(words, gr, zero_unk)
            if wd is not None:
                embeddings.append(wd)

        if len(embeddings) == 0:
            return [words[unk_string]]
        return embeddings

    def populate_embeddings(self, words, zero_unk, ngrams, scramble_rate=0):
        if ngrams:
            self.embeddings = self.populate_ngrams(self.sentence, words, zero_unk, ngrams)
        else:
            arr = self.sentence.split()
            if scramble_rate:
                if random.random() <= scramble_rate:
                    random.shuffle(arr)
            for i in arr:
                wd = lookup(words, i, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
            if len(self.embeddings) == 0:
                self.embeddings = [words[unk_string]]