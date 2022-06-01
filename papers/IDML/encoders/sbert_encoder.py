# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch.nn as nn
import logging
import warnings
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SentEncoder(nn.Module):
    def __init__(self, config_name_or_path):
        super(SentEncoder, self).__init__()
        logger.info(f"Loading Encoder @ {config_name_or_path}")
        self.bert = SentenceTransformer(config_name_or_path).to(device)
        logger.info(f"Encoder loaded.")
        self.emb_dim = 768

    def embed_sentences(self, sentences: List[str]):
        features = self.bert.tokenize(sentences)
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(device)

        out_features = self.bert.forward(features)
        embeddings = out_features['sentence_embedding']
        return embeddings

    def forward(self, sentences: List[str]):
        return self.embed_sentences(sentences)
