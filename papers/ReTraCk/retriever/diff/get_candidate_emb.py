# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import argparse
import json
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler

from retriever.schema_retriever.dense_retriever.blink.biencoder.biencoder import BiEncoderRanker
import retriever.schema_retriever.dense_retriever.blink.biencoder.data_process as data
import retriever.schema_retriever.dense_retriever.blink.candidate_ranking.utils as utils
from retriever.schema_retriever.dense_retriever.blink.common.params import BlinkParser


def load_entity_dict(logger, params):
    #path = "/home/zhiwyu/freebase_type/relation.jsonl"
    path=params["cand_to_encode"]
    entity_list = []
    logger.info("Loading entity description from path: " + path)
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list


def get_candidate_pool_tensor(
        entity_desc_list,
        tokenizer,
        max_seq_length,
        logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = []
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = data.get_candidate_representation(
            entity_text,
            tokenizer,
            max_seq_length,
            title,
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


def encode_candidate(
        reranker,
        candidate_pool,
        encode_batch_size,
        silent):
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
):
    candidate_pool = None
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(logger, params)
        candidate_pool = get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger)

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool


def main(new_config):
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    logger = utils.get_logger()
    for i in new_config:
        params[i]=new_config[i]  
    print (params)
    logger = utils.get_logger()

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer

    cand_encode_path = params.get("cand_encode_path", None)

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    )

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)

if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)

