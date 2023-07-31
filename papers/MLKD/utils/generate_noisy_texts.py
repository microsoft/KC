
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import OrderedDict, Counter
import torch

def build_vocab(examples):
    """
    Input:
        examples: [[str]]. List of examples. Each example is a list of words.
    Return:
        vocab: OrderedDict.
            key: str
            value: int
    """
    cnt = Counter()

    for sent in examples:
        words = [e.lower() for e in sent]
        cnt.update(words)

    sort_by_freq = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    vocab = OrderedDict(sort_by_freq)

    return vocab


def read_examples(fn, use_tokenizer=True):
    """
    Input:
        fn: str. Filename. Each line corresponds to a sentence.
        tokenizer_model: str. choices of ['spacy']
    Return:
        examples: [[str]]. List of examples. Each example is a list of words or tokens (if tokenizer_model is not None).
    """
    print(f'Reading examples from {fn}...')
    if use_tokenizer:
        print(f'Load basic_english tokenizer...')
        from torchtext.data import get_tokenizer
        tokenizer = get_tokenizer("basic_english")
    examples = []
    with open(fn, 'r', encoding='utf-8') as fr:
        for line in fr:
            if not use_tokenizer:
                line = line.strip().split()
            else:
                line = line.strip() 
                # tokenizer
                line = tokenizer(line)
                # convert ids to tokens
            examples.append(line)

    print(f'  Done. Number of Examples: {len(examples)}')
    return examples


def write_examples_to_file(examples, fn):
    """
    Input:
        examples: [[str]]. List of examples. Each example is a list of words.
    """
    print(f'Writing examples to {fn}...')
    with open(fn, 'w', encoding='utf-8') as fw:
        for sent in examples:
            fw.write(' '.join(sent) + '\n')
    print('  Done.')

 
def corrupt(examples, vocab, p_corruption, n_times, strategy='uniform'):
    """
    Input:
        examples: [[str]]. List of examples. Each example is a list of words.
        vocab: OrderedDict.
            key: str
            value: int
        p_corruption: float. [0, 1]. The probability that each word is replaced by another word.
        n_times: int. Controal the number of total generated/corrupted examples.
            e.g, len(corrupted_examples) = n_times * len(examples)
        strategy: str. Choice from ['uniform', 'unigram', 'uniroot']
            For details, please check https://arxiv.org/pdf/1912.12800.pdf.
    Return:
        corrupted_examples: [[str]]. Corrupted examples.
    """
    print(f'Number of input examples: {len(examples)}')
    print(f'Generate corrupted examples. p_corruption: {p_corruption}. n_times: {n_times}. strategy: {strategy}')
    word_list = list(vocab.keys())
    freq_list = torch.tensor(list(vocab.values()), dtype=torch.float)
    if strategy == 'uniroot':
        freq_list = torch.sqrt(freq_list)
    if strategy == 'power':
        freq_list = freq_list * freq_list
    freq_list /= torch.sum(freq_list)

    corrupted_examples = []
    for sent in examples:
        for _ in range(n_times):
            tmp = []
            if_replace_word = (torch.rand(len(sent)) < p_corruption)
            if strategy == 'uniform':
                replacement_indices = torch.randint(low=0, high=len(word_list)-1, size=(len(sent),))
            else:
                replacement_indices = torch.multinomial(freq_list, num_samples=len(sent), replacement=True)
            if not (len(sent) == len(if_replace_word) == len(replacement_indices)):
                raise ValueError('Lengths do not match!')

            for i in range(len(sent)):
                if if_replace_word[i]:
                    tmp.append(word_list[replacement_indices[i]])
                else:
                    tmp.append(sent[i])
            corrupted_examples.append(tmp)
    print(f'  Done. Number of generated examples: {len(corrupted_examples)}')  
    return corrupted_examples

def main():
    p_corruption = 0.8
    n_times = 5
        
    for noise_type in ['unigram', 'uniroot']:
        fn = f'../OOD_dataset/ROSTD/ROSTD_train.txt'

        fn_save = f'../OOD_dataset/ROSTD/ROSTD_train_p_corruption_{p_corruption}_n_times_{n_times}_{noise_type}.txt'

        examples = read_examples(fn, use_tokenizer=True)
        vocab = build_vocab(examples)

        corrupted_examples = corrupt(examples, vocab, p_corruption, n_times, noise_type)
        write_examples_to_file(corrupted_examples, fn_save)

if __name__ == '__main__':
    main()