# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import retriever.schema_retriever.dense_retriever.blink.main_dense as main_dense
import argparse
import json

# Path for the stored BLINK models
models_path = "./FinalWQSP/Relation/epoch_14/"
config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 1300,
    "biencoder_model": models_path+"pytorch_model.bin",
    "biencoder_config": "./FinalWQSP/Relation/training_params.txt",
    "entity_catalogue": "./freebase_type/relation.jsonl",
    "entity_encoding": "./webQSPrelationEmbedding/relationNew_emb_noBatch_14.t7",
    "fast": True,
    "faiss_index": None,
    "index_path": None,
    "bert_model": "bert-base-uncased",
    "output_path": "logs/"  # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

f = open("./freebase_type/webQSP/noBatch/relationEmbedding_train.test","r")

data_to_link = json.load(f)
import retriever.schema_retriever.dense_retriever.blink.candidate_ranking.utils as utils
_, _, _, _, _, predictions, scores, = main_dense.run(args, utils.get_logger(), *models, test_data=data_to_link)

# relationtrain_predictions->train
# relationpredictions->dev
# relationtest_predictions->test

f_res = open('./noBatch/webQSPrelationNew_train_14.test','w')
for i in range(len(predictions)):
    exp={
    "predictions": "\t".join(str(i) for i in list(predictions[i])),
    "scores": '\t'.join(str(i) for i in list(scores[i])),
    }

    f_res.write(json.dumps(exp)+"\n")

f_res.close()
