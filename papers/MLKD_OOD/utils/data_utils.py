# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from cProfile import label
import os
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import numpy as np

from functools import reduce
from json import load
from pandas import read_csv
from random import seed, shuffle

def _load_data_from_file_word(fn, text_label_seperator='\t', text_only=False):
    """
    Read data from file `fn`.
    Input:
        Filename.
            Each row corresponds to a str: "word\tlabel"
            Sentences are seperated by a blank row.
    Return:
        Data dictionary.
            if text_only:
                Keys: ["text"]
                Values: [ list(str) ]
            else:
                Keys: ["tokens", "labels"]
                Values: [ list[list(str)], list[list(str)] ]
    """
    with open (fn, 'r', encoding='utf-8') as fr:
        data = {"tokens": [], "labels": []} if not text_only else {"text": []}
        text = []
        labels = []
        for i, line in enumerate(fr):
            line = line.strip()
            if len(line) == 0:
                if len(text) > 0:
                    if text_only:
                        data['text'].append(' '.join(text))
                    else:
                        data['tokens'].append(text)
                        data['labels'].append(labels)
                    text = []
                    labels = []
                else:
                    continue
                    raise ValueError(f'multiple blank lines! Line No.: {i}')
            else:
                w, l = line.split(text_label_seperator)
                text.append(w)
                labels.append(l)

        return data

def load_raw_dataset_word(args, logger, task='OOD'):
    """
    Input:
        task:   load data for which task.
                supported choices: ['OOD', 'NER']
    Return:
        DatasetDict()
    """
    text_only_configs = {
        'OOD': True,
        'NER': False
    }
    if task not in text_only_configs.keys():
        raise ValueError('Unknown task!')

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # Load raw_datasets
    raw_datasets = DatasetDict()
    label_lists_all = []
    if args.train_file is not None:
        logger.info(f'Loading dataset from {args.train_file}...')
        train_data = _load_data_from_file_word(args.train_file, text_only=text_only_configs[task])
        raw_datasets['train'] = Dataset.from_dict(train_data)

        _, label_column_name = raw_datasets["train"].column_names
        label_lists_all.append(get_label_list(raw_datasets["train"][label_column_name]))
    if args.validation_file is not None:
        logger.info(f'Loading dataset from {args.validation_file}...')
        validation_data = _load_data_from_file_word(args.validation_file, text_only=text_only_configs[task])
        raw_datasets['validation'] = Dataset.from_dict(validation_data)

        _, label_column_name = raw_datasets["validation"].column_names
        label_lists_all.append(get_label_list(raw_datasets["validation"][label_column_name]))
    if args.test_file is not None:
        logger.info(f'Loading dataset from {args.test_file}...')
        test_data = _load_data_from_file_word(args.test_file, text_only=text_only_configs[task])
        raw_datasets['test'] = Dataset.from_dict(test_data)

        _, label_column_name = raw_datasets["test"].column_names
        label_lists_all.append(get_label_list(raw_datasets["test"][label_column_name]))

    # check label_list consistency
    label_list = label_lists_all[0]
    n_labels = len(label_list)
    for i in range(1, len(label_lists_all)):
        if len(label_lists_all[i]) != n_labels:
            raise ValueError('Label lists do not match among different dataset splits!')
        for l1, l2 in zip(label_list, label_lists_all[i]):
            if l1 != l2:
                raise ValueError('Labels do not match among different dataset splits!')

    return raw_datasets, label_list

def tokenize_dataset(args, raw_datasets, tokenizer, label_list):
    text_column_name, label_column_name= raw_datasets["validation"].column_names
    label_to_id = {l: i for i, l in enumerate(label_list)}

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on dataset",
    )

    return processed_raw_datasets


DATA_CONFIG = {
    'hf_datasets': ['imdb', 'hans', 'snli'],
    'hf_glue_datasets': ['sst2', 'mnli', 'rte', 'qnli'],
    'text_keys': {
        # 'imdb': ('text', None),
        # 'yelp_polarity': ('text', None),
        'sst2': ('sentence', None),
        'hans': ('premise', 'hypothesis'),
        'rte': ('sentence1', 'sentence2'),
        'mnli': ('premise', 'hypothesis'),
        'snli': ('premise', 'hypothesis'),
        'qnli': ('question', 'sentence'),
        './dataset/OOD/news-category/ID/train.txt': ('text', None),
        './dataset/OOD/news-category/ID/validation.txt': ('text', None),
        './dataset/OOD/news-category/OOD/validation.txt': ('text', None),
        './dataset/clinc150_train.txt': ('text', None),
        './dataset/clinc150_test.txt': ('text', None),
        './dataset/clinc150_oos_test.txt': ('text', None),
    },
    'label_key': {
        'mnli': 'label',
        'snli': 'label',
        'qnli': 'label',
        './dataset/OOD/news-category/ID/train.txt': 'label',
        './dataset/OOD/news-category/ID/validation.txt': 'label',
        './dataset/clinc150_train.txt': None,
        './dataset/clinc150_test.txt': None,
        './dataset/clinc150_oos_test.txt': None,
    },
    'eval_split':{
        'snli': 'validation',
        'mnli': 'validation_matched',
        'qnli': 'validation',
    },
    'test_split': {
        'snli': 'test',
        'mnli': 'matched_test',
        'qnli': 'test',
    },
    'idx_keys':{
        'qnli': 'idx',
        './dataset/OOD/news-category/ID/train.txt': None,
        './dataset/OOD/news-category/ID/validation.txt': None,
        './dataset/OOD/news-category/OOD/validation.txt': None,
        './dataset/clinc150_train.txt': None,
        './dataset/clinc150_test.txt': None,
        './dataset/clinc150_oos_test.txt': None,
    }
}

def load_dataset_and_tokenize(tokenizer, train_data, validation_data, test_data, max_length, padding, cache_dir, seed):
    tokenized_datasets = DatasetDict()
    label_list = None
    if train_data is not None:
        dataset = _load_dataset(train_data, 'train', cache_dir, seed)
        tokenized_datasets['train'], label_list = _tokenize_dataset(dataset, tokenizer, max_length, padding, \
                text_keys=DATA_CONFIG['text_keys'].get(train_data, ('text', None)),
                label_key=DATA_CONFIG['label_key'].get(train_data, None),
                idx_column_name=DATA_CONFIG['idx_keys'].get(train_data, None))
    if validation_data is not None:
        if ":" in validation_data:
            validation_data, split = validation_data.split(":")
        else:
            split = "validation"
        dataset = _load_dataset(validation_data, split, cache_dir, seed)
        tokenized_datasets['validation'], _ = _tokenize_dataset(dataset, tokenizer, max_length, padding, \
                text_keys=DATA_CONFIG['text_keys'].get(validation_data, ('text', None)),
                label_key=DATA_CONFIG['label_key'].get(validation_data, None),
                idx_column_name=DATA_CONFIG['idx_keys'].get(validation_data, None))
        if label_list is None:
            label_list = _
    if test_data is not None:
        test_d = []
        for test in test_data.split(','):
            if ":" in test:
                test, split = test.split(":")
            else:
                split = "test"
            dataset = _load_dataset(test, split, cache_dir, seed)
            tmp, _ = _tokenize_dataset(dataset, tokenizer, max_length, padding, \
                    text_keys=DATA_CONFIG['text_keys'].get(test, ('text', None)),
                    idx_column_name=DATA_CONFIG['idx_keys'].get(test, None))
            test_d.append(tmp)
        tokenized_datasets['test'] = test_d
    print(f'Label_list: {label_list}')
    return tokenized_datasets, label_list

def _load_dataset(data_name, split, cache_dir, seed):
    """
        Load data as Huggingface Dataset().
        Input:
            data_name:  str. name of HF dataset or local data path.
            split:      str. Split of the loaded datasets.
        Return:
            Dataset()
    """
    print(f'Loading data from [{data_name}]...')
    if data_name in DATA_CONFIG['hf_datasets']:
        dataset = load_dataset(data_name, cache_dir=cache_dir)
    elif data_name in DATA_CONFIG['hf_glue_datasets']:
        dataset = load_dataset('glue', data_name, cache_dir=cache_dir)
    else:
        # load from file
        if not os.path.exists(data_name):
            data_name = data_name.replace(f"_{seed}", "")

        raw_data = __load_data_from_file_sentence(data_name)
        dataset = Dataset.from_dict(raw_data)
    
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError('Invalid data split!')
        else:
            print(f'    Get the target split: {split}')
            dataset = dataset[split]

    print(f'    Finished. Number of examples: {len(dataset)}')
    return dataset

def __load_data_from_file_sentence(fn, text_label_seperator='\t|\t'):
    """
    Read data from file `fn`.
    Input:
        Filename.
            Each row corresponds to a sentence: "text`text_label_seperator`label".
    Return:
        Dictionary for data information.
            text:   str
            label:  str (optional)
    """
    if DATA_CONFIG['label_key'].get(fn, None) is not None:
        data = {'text': [], 'label': []}
        with open (fn, 'r', encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                line = line.strip().split(text_label_seperator)
                if len(line) < 2:
                    print(f'! ! ! Invalid text: Line #{i}')
                    continue
                    raise ValueError('Invalid row occurs!')
                data['text'].append(line[0].strip())
                data['label'].append(line[1].strip())
    else:
        data = {'text': []}
        with open (fn, 'r', encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                line = line.strip()
                if len(line) > 0:
                    data['text'].append(line)

    return data

def _tokenize_dataset(dataset, tokenizer, max_length, padding, text_keys, label_key=None, idx_column_name=None):
    if label_key is not None:
        label_list = list(set(dataset[label_key]))
        label_list.sort()
    else:
        label_list = None
    sentence1_key, sentence2_key = text_keys
    remove_column_names = dataset.column_names
    if idx_column_name is not None:
        if idx_column_name in remove_column_names:
            remove_column_names.remove(idx_column_name)
        else:
            raise ValueError(f'Invalid [idx_column_name]: {idx_column_name}')
        
    print(f'    label_list: {label_list}')
    print(f'    text_keys: {text_keys}')
    print(f'    label_key: {label_key}')
    print(f'    idx_column_name: {idx_column_name}')

    def preprocess_function(examples):
        texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))

        tokenized_inputs = tokenizer(
            *texts,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            truncation=True
        )

        if label_list is not None:
            label_to_id = {l: i for i, l in enumerate(label_list)}
            if label_list is None:
                raise ValueError('Label list should not be None.')
            tokenized_inputs["labels"] = [label_to_id[l] for l in examples[label_key]]

        return tokenized_inputs

    tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_column_names,
            desc="Running tokenizer on dataset",
        )
    
    for index in range(3):
            print(f"    Sample {index} of the training set: {tokenized_dataset[index]}.")

    return (tokenized_dataset, label_list)


def split_ood_and_save_data(folder_name, raw_datasets, cn_text, cn_label, type, type_label_set):
    """
    Input:
        folder_name:        str. Name of the folder to save the processed data.
        raw_datasets:       HuggingFace DatasetDict. keys: k1, k2, k3. e.g., train, validation, test.
        cn_text:            str. Column name of text. e.g., `headline`.
        cn_label:           str. Column name of label. e.g., `category`.
        type:               str. Choice of ['ID', 'OOD'].
        type_label_set:     [str]. labels correspond to `type`.
                                If type == 'ID', labels not in `type_label_set` will be regard as OOD examples.
    Return:
        folder_name
            |- ID
                |- train.txt (k1)           each row correspons to an exmaple: "text \t|\t label"
                |- validation.txt (k2)
                |- test.txt (k3)
            |- OOD
                |- train.txt (k1)
                |- validation.txt (k2)
                |- test.txt (k3)
            |- train.txt (a combination of `ID/train.txt` and `OOD/train.txt`) "text \t|\t ID/OOD"
            |- validation.txt (a combination of `ID/validation.txt` and `OOD/validation.txt`)
            |- test.txt (a combination of `ID/test.txt` and `OOD/test.txt`)
    """
    # check input parameters:
    if type == 'ID':
        other = 'OOD'
    elif type == 'OOD':
        other = 'ID'
    else:
        raise ValueError(f'Invalid input for `type`: {type}')

    print(f'Split datasets to ID and OOD, save to {folder_name}...')
    os.makedirs(folder_name, exist_ok=True)
    type_folder = os.path.join(folder_name, type)
    os.makedirs(type_folder, exist_ok=True)
    other_folder = os.path.join(folder_name, other)
    os.makedirs(other_folder, exist_ok=True)
    
    for k in raw_datasets.keys():
        print(f'Saveing split - {k} ...')
        ds = raw_datasets[k]
        if cn_text not in ds.column_names:
            raise ValueError(f'Invalid input column name for `cn_text`: {cn_text}!')
        if cn_label not in ds.column_names:
            raise ValueError(f'Invalid input column name for `cn_label`: {cn_label}!')

        with open(f'{type_folder}/{k}.txt', 'w', encoding='utf-8') as f_type, \
            open(f'{other_folder}/{k}.txt', 'w', encoding='utf-8') as f_other, \
                open(f'{folder_name}/{k}.txt', 'w', encoding='utf-8') as f_all:
                    for i in range(len(ds)):
                        # get values to save for each example.
                        if (i % 1000 == 0):
                            print(f'    To #{i}.')
                        # write into corresponding file
                        if ds[i][cn_label] in type_label_set:
                            f_type.write(f'{ds[i][cn_text]}\t|\t{ds[i][cn_label]}\n')
                            f_all.write(f'{ds[i][cn_text]}\t|\t{type}\n')
                        else:
                            f_other.write(f'{ds[i][cn_text]}\t|\t{ds[i][cn_label]}\n')
                            f_all.write(f'{ds[i][cn_text]}\t|\t{other}\n')

def reader_sst_datasets(input_files, output_path, sample=None):
    data = []
    for file_name in input_files:
        print(file_name)
        with open(file_name, 'r') as f:
            tmp_data = [ii.strip().split("\t")[1] for jj, ii in enumerate(f) if jj and ii.strip()]
            if sample:
                np.random.shuffle(tmp_data)
                tmp_data = tmp_data[:sample]
            data.extend(tmp_data)
    with open(output_path, 'w') as f:
        f.write('\n'.join(data))

def convert_df_txt(df, output_path):
    text = [row["text"] for ii, row in df.iterrows()]
    with open(output_path, 'w') as f:
        f.write('\n'.join(text))

def load_sst_dataset():
    train_df = load_extra_dataset("./dataset/sst/sst-train.txt", drop_index="sentence_index", label=1)
    test_df = load_extra_dataset("./dataset/sst/sst-test.txt", drop_index="sentence_index", label=1)
    convert_df_txt(train_df, "OOD_dataset/SST/sst_train.txt")
    convert_df_txt(test_df, "OOD_dataset/SST/sst_test.txt")

    for seed in [171, 354, 550, 667, 985]:
        ood_snli_df = load_extra_dataset("./dataset/sst/snli-dev.txt", drop_index="index", label=0)
        ood_rte_df = load_extra_dataset("./dataset/sst/rte-dev.txt", drop_index="index", label=0)
        ood_20ng_df = load_extra_dataset("./dataset/sst/20ng-test.txt", drop_index="index", label=0)
        ood_multi30k_df = load_extra_dataset("./dataset/sst/multi30k-val.txt", drop_index="index", label=0)
        ood_snli_df = ood_snli_df.sample(n=500, random_state=seed)
        ood_rte_df = ood_rte_df.sample(n=500, random_state=seed)
        ood_20ng_df = ood_20ng_df.sample(n=500, random_state=seed)
        ood_multi30k_df = ood_multi30k_df.sample(n=500, random_state=seed)
        ood_df = pd.concat([ood_snli_df, ood_rte_df, ood_20ng_df, ood_multi30k_df])
        convert_df_txt(ood_df, f"OOD_dataset/SST/sst_oos_test_{seed}.txt")
    

def load_extra_dataset(file_path="./dataset/SSTSentences.txt", drop_index=False, label=0):
    df = pd.read_csv(file_path, sep='\t', header=0)
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns=drop_index, inplace=True)
    df.dropna(inplace=True)
    return df

def load_cimdb():
    res = []
    for i in ["train", "dev", "test"]:
        path = f"intermediate/counterfactual-imdb/{i}_paired.tsv"
        with open(path) as f:
            data = [ii.strip().split("\t")[1].replace("<br />", " ").replace("  ", " ") for ii in f.readlines()][1:]
            res.extend(data)
    with open("OOD_dataset/ChallengeData/cimdb.txt", "w") as f:
        f.write("\n".join(res))

def load_stress_test():
    kv = [
        ("intermediate/StressTests/Antonym/multinli_0.9_antonym_matched.jsonl", "OOD_dataset/ChallengeData/Antonym.txt"), 
        ("intermediate/StressTests/Length_Mismatch/multinli_0.9_length_mismatch_matched.jsonl", "OOD_dataset/ChallengeData/Length_Mismatch.txt"),
        ("intermediate/StressTests/Negation/multinli_0.9_negation_matched.jsonl", "OOD_dataset/ChallengeData/Negation.txt"),
        ("intermediate/StressTests/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl", "OOD_dataset/ChallengeData/Numerical_Reasoning.txt"),
        ("intermediate/StressTests/Word_Overlap/multinli_0.9_taut2_matched.jsonl", "OOD_dataset/ChallengeData/Word_Overlap.txt"),
        ([
            "intermediate/StressTests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl",
            "intermediate/StressTests/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl",
            "intermediate/StressTests/Spelling_Error/multinli_0.9_dev_gram_keyboard_matched.jsonl",
            "intermediate/StressTests/Spelling_Error/multinli_0.9_dev_gram_swap_matched.jsonl",
        ], "OOD_dataset/ChallengeData/Spelling_Error.txt"),
    ]
    for k, v in kv:
        res = []
        if not isinstance(k, list):
            k = [k]
        for x in k:
            with open(x) as f:
                data = [json.loads(ii) for ii in f.readlines()]
                data = [f"{ii['sentence1']} {ii['sentence2']}" for ii in data]
                res.extend(data)
        with open(v, "w") as f:
            f.write("\n".join(res))

def load_stress_test_ood():
    kv = [
        ('intermediate/Stress Tests/Negation/multinli_0.9_negation_mismatched.jsonl', 'OOD_dataset/ChallengeData/Negation.txt'),
        ('intermediate/Stress Tests/Length_Mismatch/multinli_0.9_length_mismatch_mismatched.jsonl', 'OOD_dataset/ChallengeData/Length_Mismatch.txt'),
        ('intermediate/Stress Tests/Word_Overlap/multinli_0.9_taut2_mismatched.jsonl', 'OOD_dataset/ChallengeData/Word_Overlap.txt'),
        ("intermediate/Stress Tests/Antonym/multinli_0.9_antonym_mismatched.jsonl", "OOD_dataset/ChallengeData/Antonym.txt"),
        ('intermediate/Stress Tests/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl', 'OOD_dataset/ChallengeData/Numerical_Reasoning.txt'),
    ]
    for k, v in kv:
        sentence1s = list(map(lambda x: json.loads(x)['sentence1'], open(k, 'r').readlines()))
        sentence2s = list(map(lambda x: json.loads(x)['sentence2'], open(k, 'r').readlines()))
        with open(v, 'w') as f:
            f.write('\n'.join([f"{ii} {jj}" for ii, jj in zip(sentence1s, sentence2s)]))
    sentence1s = list(map(lambda x: json.loads(x)['sentence1'], reduce(lambda x, y: x + y, [open(x, 'r').readlines() for x in glob.glob('intermediate/Stress Tests/Spelling_Error/multinli_0.9_dev_*_mismatched.jsonl')])))
    sentence2s = list(map(lambda x: json.loads(x)['sentence2'], reduce(lambda x, y: x + y, [open(x, 'r').readlines() for x in glob.glob('intermediate/Stress Tests/Spelling_Error/multinli_0.9_dev_*_mismatched.jsonl')])))
    with open('OOD_dataset/ChallengeData/Spelling_Error.txt', 'w') as f:
        f.write('\n'.join([f"{ii} {jj}" for ii, jj in zip(sentence1s, sentence2s)]))


def load_stress_test_test():
    kv = [
        ('antonym_matched', 'OOD_dataset/ChallengeData/Antonym.txt'),
        ('dev_gram_contentword_swap_perturbed_matched', 'OOD_dataset/ChallengeData/Spelling_Error.txt'),
        ('length_mismatch_matched', "OOD_dataset/ChallengeData/Length_Mismatch.txt"),
        ('negation_matched', 'OOD_dataset/ChallengeData/Negation.txt'),
        ('quant_hard', 'OOD_dataset/ChallengeData/Numerical_Reasoning.txt'),
        ('wordoverlap_matched', 'OOD_dataset/ChallengeData/Word_Overlap.txt'),
    ]
    g = defaultdict(list)
    with open("intermediate/stress_tests.jsonl") as f:
        data = [json.loads(ii) for ii in f.readlines()]
    for ii in data:
        g[ii['source']].append(f"{ii['sentence1']} {ii['sentence2']}")
    for k, v in kv:
        with open(v, "w") as f:
            f.write("\n".join(g[k]))
    

def load_20ng_ag(dataset_name):
    train_dir = f"intermediate/{dataset_name}/train/"
    test_dir = f"intermediate/{dataset_name}/test/"
    output_dir = f"OOD_dataset/{dataset_name.replace('_od', '')}/"
    cast_list = os.listdir(train_dir)

    for case in cast_list:
        with open(f"{train_dir}{case}") as f:
            data = [ii.strip() for ii in f.readlines() if ii.strip()]
        with open(f"{output_dir}train/{case}", "w") as f:
            f.write("\n".join(data))
        with open(f"{test_dir}{case}") as f:
            data = [ii.strip() for ii in f.readlines() if ii.strip()]
        with open(f"{output_dir}test/{case}", "w") as f:
            f.write("\n".join(data))
        for other in cast_list:
            if other == case:
                continue
            with open(f"{output_dir}ood/{other}", "a") as f:
                f.write("\n".join(data))

def load_dilgue():
    def export_txt(text, output_dir):
        with open(output_dir, "w") as f:
            f.write("\n".join(text))
    full_train = read_csv('intermediate/SNIPS/train.csv').groupby('label')['text'].apply(set).to_dict()
    full_valid = read_csv('intermediate/SNIPS/valid.csv').groupby('label')['text'].apply(list).to_dict()
    full_test = read_csv('intermediate/SNIPS/test.csv').groupby('label')['text'].apply(set).to_dict()
    classes = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
    
    for s in [171, 354, 550, 667, 985]:
        seed(s)
        shuffle(classes)
        train_id = list(reduce(set.union, map(lambda x: full_train[x], classes[ : 5])))
        valid_id = list(reduce(lambda x, y: x + y, map(lambda x: full_valid[x], classes[ : 5])))
        valid_ood = list(reduce(lambda x, y: x + y, map(lambda x: full_valid[x], classes[5 : ])))
        test_id = list(reduce(set.union, map(lambda x: full_test[x], classes[ : 5])))
        test_ood = list(reduce(set.union, map(lambda x: full_test[x], classes[5 : ])))
        export_txt(train_id, f"OOD_dataset/Dialog/train/SNIPS_{s}.txt")
        export_txt(test_id, f"OOD_dataset/Dialog/test/SNIPS_{s}.txt")
        export_txt(test_ood, f"OOD_dataset/Dialog/ood/SNIPS_{s}.txt")

    train_id = list(map(lambda x: x.split('\t')[2].strip(), open('intermediate/multilingual_task_oriented_dialog_slotfilling/en/train-en.tsv', 'r').readlines()))
    test_id = list(map(lambda x: x.split('\t')[2].strip(), open('intermediate/multilingual_task_oriented_dialog_slotfilling/en/test-en.tsv', 'r').readlines()))

    export_txt(train_id, f"OOD_dataset/Dialog/train/ROSTD.txt")
    export_txt(test_id, f"OOD_dataset/Dialog/test/ROSTD.txt")
    
    ood = list(map(lambda x: x.split('\t')[2].strip(), open('intermediate/OODrelease.tsv', 'r').readlines()))
    for s in [171, 354, 550, 667, 985]:
        seed(s)
        shuffle(ood)
        test_ood = ood[1500 : ]
        export_txt(test_ood, f"OOD_dataset/Dialog/ood/ROSTD_{s}.txt")


    raw = list(zip([x.strip() for x in open('intermediate/STC2/StackOverflow.txt').readlines()], [int(x.strip()) for x in open('intermediate/STC2/StackOverflow_gnd.txt').readlines()]))
    full = [set(map(lambda x: x[0], filter(lambda x: x[1] == i, raw))) for i in range(1, 21)]
    partition = []
    seed(3)
    for c in full:
        texts = list(c)
        shuffle(texts)
        four_par = len(texts) // 25
        partition.append((texts[ : four_par], texts[four_par : four_par * 2], texts[four_par * 2 : ])) # test/validation/train parts
    for s in [171, 354, 550, 667, 985]:
        seed(s)
        shuffle(partition)
        train_id = list(reduce(lambda x, y: x + y, map(lambda x: x[2], partition[ : 11])))
        test_id = list(reduce(lambda x, y: x + y, map(lambda x: x[0], partition[ : 11])))
        test_ood = list(reduce(lambda x, y: x + y, map(lambda x: x[0], partition[11 : 17])))
        export_txt(train_id, f"OOD_dataset/Dialog/train/Stackoverflow_{s}.txt")
        export_txt(test_id, f"OOD_dataset/Dialog/test/Stackoverflow_{s}.txt")
        export_txt(test_ood, f"OOD_dataset/Dialog/ood/Stackoverflow_{s}.txt")


if __name__ == '__main__':
    # raw_datasets = load_dataset('Fraser/news-category-dataset')
    # values = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY']
    # split_ood_and_save_data('dataset/OOD/news-category', raw_datasets, 'headline', 'category', 'ID', values)
    
    # j = 0
    # cn_label = 'category'
    # ds = raw_datasets['train']
    # for i in range(len(ds)):
    #     if ds[i][cn_label] in values:
    #         j += 1
    #         if j > 1024:
    #             print(ds[i])


    # raw_datasets = load_dataset('dbpedia_14')
    # values = [0, 1, 2, 3]
    # split_ood_and_save_data('dataset/OOD/news-category', raw_datasets, ['content', "labels"], "labels", values)
    pass
