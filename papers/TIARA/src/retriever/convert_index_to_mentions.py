# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    return parser.parse_args()


def convert(split):
    base_file = f'el_files/{split}_schema10with_top5LFs_bootleg_prior_entities_top5_ent.json'
    with open(base_file, 'r', encoding='utf-8') as f_grail_gt:
        GrailEL = json.load(f_grail_gt)
    # dev_schema10with_top5LFs_bootleg_prior_entities_top5_ent.json
    # test_dense_v0_bootleg_prior_entities_top5_ent.json

    index_file = f'el_files/SpanIndex_{split}.txt'
    with open(index_file, 'r', encoding='utf-8') as f_span:
        Mention_span = json.load(f_span)

    output_file = f'el_files/SpanMD_{split}.json'
    f_mentions = open(output_file, 'w', encoding='utf-8')
    span_mentions = {}
    total_mentions = 0

    for i in range(len(GrailEL)):
        mentions = []

        for mention in Mention_span[i]:
            index1, index2 = mention[0], mention[1]
            m_lis = GrailEL[i]["bert_tokens"][index1:index2 + 1]
            m = ' '.join(m_lis)
            mentions.append(m)
            total_mentions += 1

        mentions.sort(key=lambda i: len(str(i)), reverse=True)
        span_mentions[str(GrailEL[i]["qid"])] = mentions

    json_str = json.dumps(span_mentions)
    f_mentions.write(json_str)
    print('number of mentions: {}'.format(total_mentions))


if __name__ == '__main__':
    args = _parse_args()
    convert(args.split)
