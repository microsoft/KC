import logging
import warnings
import torch
import torch.nn as nn
from argparse import Namespace

from sacremoses import MosesTokenizer
from encoders.para_models import load_model, embed, batcher

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ParaEncoder(nn.Module):
    def __init__(self, load_file, sp_model):
        super(ParaEncoder, self).__init__()
        logger.info(f"Loading Encoder @ {load_file}-{sp_model}")
        if torch.cuda.is_available():
            gpu = 1
        else:
            gpu = 0
        model_args = Namespace(load_file=load_file, sp_model=sp_model, gpu=gpu)
        self.model, _ = load_model(None, model_args)
        self.entok = MosesTokenizer(lang='en')
        self.model.eval()
        self.new_args = Namespace(batch_size=32, entok=self.entok, sp=self.model.sp,
                             params=model_args, model=self.model, lower_case=self.model.args.lower_case,
                             tokenize=self.model.args.tokenize)
        self.emb_dim = 1024
        logger.info(f"Encoder loaded.")

    def embed_sentences(self, sentences):
        self.model.eval()
        np_embedding = embed(self.new_args, batcher, sentences)
        return torch.tensor(np_embedding).to(device)

    def forward(self, sentences):
        return self.embed_sentences(sentences)