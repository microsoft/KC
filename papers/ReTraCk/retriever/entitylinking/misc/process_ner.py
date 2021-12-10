# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from retriever.entitylinking.bert_ner.bert import Ner
import json


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


# model = Ner("out_base_uncased_WebQSP_distant_year_fix_offset/")
model = Ner("out_base_uncased_GrailQA_datetime/")


def process_one_file(in_fn, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in fp:
                sent = json.loads(line.strip())
                # question = " ".join(sent["tokenized_question"])
                question = sent["sentence"]
                output = model.predict(question)
                ner_output = get_entities([w['tag'] for w in output])
                sent["ner_output"] = []
                sent["bert_tokens"] = [w['word'] for w in output]
                for x in ner_output:
                    sent["ner_output"].append((x[0], x[1], x[2] + 1))
                re_fp.write("{}\n".format(json.dumps(sent)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--infn", type=str, default=r"F:\Workspace\BERT-NER\WebQDistantData\webqsp_dev_part_aligned.jsonl")
    # parser.add_argument("--outfn", type=str, default=r"F:\Workspace\BERT-NER\WebQDistantData\ner\webqsp_dev_part_v1.json")
    parser.add_argument("--infn", type=str,
                        default=r"E:\users\v-shuanc\Workspace-GCR146\F\Workspace\GrailQA\Data\DistantData\test_bootleg_data.jsonl")
    parser.add_argument("--outfn", type=str,
                        default=r"E:\users\v-shuanc\Workspace-GCR146\F\Workspace\GrailQA\Data\DistantData\test_bootleg_data_again.jsonl")
    args = parser.parse_args()
    process_one_file(args.infn, args.outfn)
