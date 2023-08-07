#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import gc
import io
import json
import logging
import math
import os
import pickle as pkl
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import LanguageModelingArgs
from simpletransformers.custom_models.models import (
    ElectraForLanguageModelingModel, ElectraForPreTraining)
from simpletransformers.language_modeling.language_modeling_utils import (
    DocumentDataset, SimpleDataset, calculate_acc, get_metrics, mask_tokens,
    mask_tokens_vanilla, merge_batches, mp_score, neg_entropy, pl_score,
    plot_confusion_matrix, plot_to_image)
from simpletransformers.language_modeling.ood_metrics import compute_all_scores
from scikitplot.metrics import plot_roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay, auc,
                             average_precision_score, confusion_matrix,
                             label_ranking_average_precision_score,
                             matthews_corrcoef, mean_squared_error,
                             precision_recall_curve, roc_curve)
from tensorboardX import SummaryWriter
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, AdamW
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
# from tqdm.auto import tqdm, trange
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AutoConfig, AutoModelWithLMHead,
                          AutoTokenizer, BertConfig, BertForMaskedLM,
                          BertTokenizer, CamembertConfig, CamembertForMaskedLM,
                          CamembertTokenizer, DistilBertConfig,
                          DistilBertForMaskedLM, DistilBertTokenizer,
                          ElectraConfig, ElectraForMaskedLM, ElectraTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          LongformerConfig, LongformerForMaskedLM,
                          LongformerTokenizer, OpenAIGPTConfig,
                          OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          PreTrainedModel, PreTrainedTokenizer, RobertaConfig,
                          RobertaForMaskedLM, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.data.datasets.language_modeling import (
    LineByLineTextDataset, TextDataset)
from transformers.modeling_electra import ElectraEmbeddings

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModelWithLMHead, AutoTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "distilbert":
    (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "electra":
    (ElectraConfig, ElectraForLanguageModelingModel, ElectraTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "longformer":
    (LongformerConfig, LongformerForMaskedLM, LongformerTokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
}


class LanguageModelingModel:
    def __init__(
        self,
        model_type,
        model_name,
        masks,
        generator_name=None,
        discriminator_name=None,
        train_files=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a LanguageModelingModel.
        Args:
            model_type: The type of model (gpt2, openai-gpt, bert, roberta, distilbert, camembert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            generator_name (optional): A pretrained model name or path to a directory containing an ELECTRA generator model.
            discriminator_name (optional): A pretrained model name or path to a directory containing an ELECTRA discriminator model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            train_files (optional): List of files to be used when training the tokenizer.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        self.args = self._load_model_args(model_name)
        self.extra_args = args

        self.current_epoch = 0

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            self.CELoss = nn.CrossEntropyLoss()

        self.masks = masks  # should be in the if
        logger.info(f'# MASKS: {len(self.masks)}')

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LanguageModelingArgs):
            self.args = args

        if "sweep_config" in kwargs:
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {
                key: value["value"]
                for key, value in sweep_config.as_dict().items()
                if key != "_wandb"
            }
            self.args.update_from_dict(sweep_values)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.local_rank != -1:
            logger.info(f"local_rank: {self.args.local_rank}")
            torch.distributed.init_process_group(backend="nccl")
            cuda_device = self.args.local_rank

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False.")
        else:
            self.device = "cpu"

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        self.args.model_name = model_name
        self.args.model_type = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer_class = tokenizer_class
        new_tokenizer = False

        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.args.tokenizer_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name:
            if self.args.model_name == "electra":
                self.tokenizer = tokenizer_class.from_pretrained(
                    generator_name, cache_dir=self.args.cache_dir, **kwargs)
                self.args.tokenizer_name = self.args.model_name
            else:
                self.tokenizer = tokenizer_class.from_pretrained(
                    model_name, cache_dir=self.args.cache_dir, **kwargs)
                self.args.tokenizer_name = self.args.model_name
        else:
            if not train_files:
                raise ValueError(
                    "model_name and tokenizer_name are not specified."
                    "You must specify train_files to train a Tokenizer.")
            else:
                print("train_files", train_files)
                self.train_tokenizer(train_files)
                new_tokenizer = True
        if self.args.config_name:
            self.config = config_class.from_pretrained(
                self.args.config_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name and self.args.model_name != "electra":
            self.config = config_class.from_pretrained(
                model_name, cache_dir=self.args.cache_dir, **kwargs)
        else:
            self.config = config_class(**self.args.config, **kwargs)
        if self.args.vocab_size:
            self.config.vocab_size = self.args.vocab_size
        if new_tokenizer:
            self.config.vocab_size = len(self.tokenizer)

        if self.args.model_type == "electra":
            if generator_name:
                self.generator_config = ElectraConfig.from_pretrained(
                    generator_name)
            elif self.args.model_name:
                self.generator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args.model_name, "generator_config"),
                    **kwargs,
                )
            else:
                self.generator_config = ElectraConfig(
                    **self.args.generator_config, **kwargs)
                if new_tokenizer:
                    self.generator_config.vocab_size = len(self.tokenizer)

            if discriminator_name:
                self.discriminator_config = ElectraConfig.from_pretrained(
                    discriminator_name)
            elif self.args.model_name:
                self.discriminator_config = ElectraConfig.from_pretrained(
                    os.path.join(self.args.model_name, "discriminator_config"),
                    **kwargs,
                )
            else:
                self.discriminator_config = ElectraConfig(
                    **self.args.discriminator_config, **kwargs)
                if new_tokenizer:
                    self.discriminator_config.vocab_size = len(self.tokenizer)

        if self.args.block_size <= 0:
            self.args.block_size = min(self.args.max_seq_length,
                                       self.tokenizer.max_len)
        else:
            self.args.block_size = min(self.args.block_size,
                                       self.tokenizer.max_len,
                                       self.args.max_seq_length)

        if self.args.model_name:
            if self.args.model_type == "electra":
                if self.args.model_name == "electra":
                    generator_model = ElectraForMaskedLM.from_pretrained(
                        generator_name)
                    discriminator_model = ElectraForPreTraining.from_pretrained(
                        discriminator_name)
                    self.model = ElectraForLanguageModelingModel(
                        config=self.config,
                        output_size=len(self.masks),
                        generator_model=generator_model,
                        discriminator_model=discriminator_model,
                        generator_config=self.generator_config,
                        discriminator_config=self.discriminator_config,
                        tie_generator_and_discriminator_embeddings=self.args.
                        tie_generator_and_discriminator_embeddings,
                        random_generator=self.extra_args['random_generator'])
                    model_to_resize = (self.model.generator_model.module
                                       if hasattr(self.model.generator_model,
                                                  "module") else
                                       self.model.generator_model)
                    model_to_resize.resize_token_embeddings(len(
                        self.tokenizer))

                    model_to_resize = (self.model.discriminator_model.module if
                                       hasattr(self.model.discriminator_model,
                                               "module") else
                                       self.model.discriminator_model)
                    model_to_resize.resize_token_embeddings(len(
                        self.tokenizer))
                    self.model.generator_model = generator_model
                    self.model.discriminator_model = discriminator_model
                else:
                    self.model = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        cache_dir=self.args.cache_dir,
                        generator_config=self.generator_config,
                        discriminator_config=self.discriminator_config,
                        **kwargs,
                    )
                    self.model.load_state_dict(
                        torch.load(
                            os.path.join(self.args.model_name,
                                         "pytorch_model.bin")))
            else:
                self.model = model_class.from_pretrained(
                    model_name,
                    config=self.config,
                    cache_dir=self.args.cache_dir,
                    **kwargs,
                )
        else:
            logger.info(" Training language model from scratch")
            if self.args.model_type == "electra":
                generator_model = ElectraForMaskedLM(
                    config=self.generator_config)
                discriminator_model = ElectraForPreTraining(
                    config=self.discriminator_config,
                    extra_args=self.extra_args)

                self.model = ElectraForLanguageModelingModel(
                    config=self.config,
                    output_size=len(self.masks),
                    extra_args=self.extra_args,
                    generator_model=generator_model,
                    discriminator_model=discriminator_model,
                    generator_config=self.generator_config,
                    discriminator_config=self.discriminator_config,
                    tie_generator_and_discriminator_embeddings=self.args.
                    tie_generator_and_discriminator_embeddings,
                    random_generator=self.extra_args['random_generator'])
                model_to_resize = (self.model.generator_model.module
                                   if hasattr(self.model.generator_model,
                                              "module") else
                                   self.model.generator_model)
                model_to_resize.resize_token_embeddings(len(self.tokenizer))

                model_to_resize = (self.model.discriminator_model.module
                                   if hasattr(self.model.discriminator_model,
                                              "module") else
                                   self.model.discriminator_model)
                model_to_resize.resize_token_embeddings(len(self.tokenizer))
            else:
                self.model = model_class(config=self.config)
                model_to_resize = self.model.module if hasattr(
                    self.model, "module") else self.model
                model_to_resize.resize_token_embeddings(len(self.tokenizer))

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion.")
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

    def train_model(
        self,
        train_file,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_file=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_file'
        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.
        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_file is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_file is not specified."
                " Pass eval_file to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(
                output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(
                    output_dir))

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_file,
                                                     verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, tr_loss = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_file=eval_file,
            verbose=verbose,
            **kwargs,
        )

        self._save_model(output_dir, model=self.model)
        if self.args.model_type == "electra":
            self.save_discriminator()
            self.save_generator()

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(
                self.args.model_type, output_dir))

    def build_text_samples(self, inlier_file, outlier_file):
        self.lines = []
        with open(inlier_file) as fp:
            line = fp.readline()
            while line:
                if line != '\n':
                    self.lines.append(line)
                line = fp.readline()
        self.inlier_len = len(self.lines)
        with open(outlier_file) as fp:
            line = fp.readline()
            while line:
                if line != '\n':
                    self.lines.append(line)
                line = fp.readline()
        self.outlier_len = len(self.lines) - self.inlier_len
        logger.info(
            f" [LM CLASS - EVAL] inliers: {self.inlier_len} / outliers: {self.outlier_len}"
        )

    def train_model_anomaly(
        self,
        train_file,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_file=None,
        eval_file_outlier=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_file'
        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.
        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update(args)

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        if "train_document" in self.extra_args and self.extra_args["train_document"] == True:
            train_dataset = DocumentDataset(
                self.tokenizer,
                self.args,
                train_file,
                self.args.block_size,
                2,
                sliding_window=self.args.sliding_window)

            eval_dataset = DocumentDataset(
                self.tokenizer,
                self.args,
                eval_file,
                self.args.block_size,
                2,
                sliding_window=self.args.sliding_window)
            logger.info(" Using train_document")
        else:
            train_dataset = SimpleDataset(
                self.tokenizer,
                self.args,
                train_file,
                self.args.block_size,
                2,
                sliding_window=self.args.sliding_window)

            eval_dataset = SimpleDataset(
                self.tokenizer,
                self.args,
                eval_file,
                self.args.block_size,
                2,
                sliding_window=self.args.sliding_window)

        eval_doc_dataset = DocumentDataset(
            self.tokenizer,
            self.args,
            eval_file,
            self.args.block_size,
            2,
            sliding_window=self.args.sliding_window)

        eval_doc_dataset_outliers = DocumentDataset(
            self.tokenizer,
            self.args,
            eval_file_outlier,
            self.args.block_size,
            2,
            sliding_window=self.args.sliding_window)

        self.build_text_samples(eval_file, eval_file_outlier)

        global_step, tr_loss = self.train_anomaly(
            train_dataset,
            eval_dataset,
            eval_doc_dataset,
            eval_doc_dataset_outliers,
            show_running_loss=show_running_loss,
            verbose=verbose,
            **kwargs,
        )

        self._save_model(output_dir, model=self.model)
        if self.args.model_type == "electra":
            self.save_discriminator()
            self.save_generator()

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(
                self.args.model_type, output_dir))

    def extract_representations(self,
                                dataloader,
                                step,
                                path='./representations/',
                                name='train',
                                dump=True):
        with torch.no_grad():
            total = 0

            representations = []
            model = self.model.discriminator_model
            model.eval()

            batches = merge_batches(dataloader, f'repr_extract_{name}_rand')
            for batch, idx_list in batches:
                if batch is None:
                    continue
                batch = batch.to(self.device).long()
                output = model(batch)
                reprs = output[-1].cpu().numpy()
                reprs = np.split(reprs, idx_list)
                reprs = np.array([np.mean(t, axis=0) for t in reprs])
                representations.append(reprs)
            del batches

            representations = np.concatenate(representations, axis=0)

            dump_name = self.extra_args['tensorboard_dir'][10:]

            if dump:
                with open(f'{path}{name}{dump_name}_{step}.pkl',
                          'wb') as f_out:
                    pkl.dump(representations, f_out)
            del representations

    def test_anomaly(self, batch):
        args = self.args
        tokenizer = self.tokenizer
        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            masks = self.masks
            masks_len = len(masks)
            unsorted_outs = []
        else:
            masks_len = self.extra_args['vanilla_electra']['no_masks']

        rmd_classes = masks_len #hardcoded, should be changed for RMD anomaly detection
        masks_len = 1

        unsorted_outs_rtd = []
        repr_list = []
        preds = []
        batch_size = batch.shape[0]
        unsorted_outs_rtd_electra = []
        binary_labels = []
        pad_labels = []

        model = self.model
        model.eval()

        input_tokens = []

        if 'replace_tokens' in self.extra_args:
            replace_tokens = self.extra_args['replace_tokens']
        else:
            replace_tokens = True

        with torch.no_grad():

            if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                # to do: just viable masks
                for mask_idx, mask_ in enumerate(masks):

                    mask_['mask'] = [False for _ in mask_['mask']]

                    inputs, labels, _ = mask_tokens(batch, tokenizer, masks, args, custom_mask=mask_, train=False, no_mask=True)

                    is_pad = [[1 if el == tokenizer.convert_tokens_to_ids(tokenizer.pad_token) else 0 for el in input_] for input_ in inputs]
                    is_pad = np.array(is_pad)
                    pad_labels.append(np.expand_dims(is_pad, axis=1))

                    labels_bin = [[0 if el == -100 else 1 for el in mask]
                                  for mask in labels]

                    labels_bin = np.array(labels_bin)
                    binary_labels.append(np.expand_dims(labels_bin, axis=1))

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    output = model(inputs, masked_lm_labels=labels, replace_tokens=replace_tokens) if args.mlm else model(inputs, labels=labels)

                    d_output, d_inputs, rtd_output = output[2], output[3].cpu(), output[8]

                    representations = output[7].cpu()

                    out_softmax = F.softmax(d_output).cpu().numpy()

                    unsorted_outs.append(out_softmax)
                    repr_list.append(representations)

                    rtd_output = F.sigmoid(rtd_output).cpu().numpy()

                    rtd_full_probs = 1 - rtd_output
                    rtd_full_probs = np.expand_dims(rtd_full_probs, axis=-1)

                    rtd_full_probs = np.concatenate((rtd_full_probs, np.expand_dims(rtd_output, axis=-1)), axis=-1)
                    unsorted_outs_rtd_electra.append(np.expand_dims(rtd_full_probs, axis=1))

                    if len(rtd_output.shape) == 1:
                        rtd_output = np.expand_dims(rtd_output, 0)

                    row_sums_rtd = rtd_output.sum(axis=1)
                    rtd_output_normalized = rtd_output / row_sums_rtd[:, np.newaxis]
                    unsorted_outs_rtd.append(rtd_output_normalized)
                    break
            else:
                for mask_idx in range(self.extra_args['vanilla_electra']['no_masks']):

                    inputs, labels = mask_tokens_vanilla(
                        batch, tokenizer, args) if args.mlm else (batch, batch)

                    is_pad = [[1 if el == tokenizer.convert_tokens_to_ids(tokenizer.pad_token) else 0 for el in input_] for input_ in inputs]
                    is_pad = np.array(is_pad)
                    pad_labels.append(np.expand_dims(is_pad, axis=1))

                    labels_bin = [[0 if el == -100 else 1 for el in mask]
                                  for mask in labels]

                    labels_bin = np.array(labels_bin)
                    binary_labels.append(np.expand_dims(labels_bin, axis=1))

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    output = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)

                    rtd_output = output[6]
                    rtd_output = F.sigmoid(rtd_output).cpu().numpy()

                    rtd_full_probs = 1 - rtd_output
                    rtd_full_probs = np.expand_dims(rtd_full_probs, axis=-1)
                    rtd_full_probs = np.concatenate((rtd_full_probs, np.expand_dims(rtd_output, axis=-1)), axis=-1)
                    unsorted_outs_rtd_electra.append(np.expand_dims(rtd_full_probs, axis=1))

                    row_sums_rtd = rtd_output.sum(axis=1)
                    rtd_output_normalized = rtd_output / row_sums_rtd[:, np.newaxis]
                    unsorted_outs_rtd.append(rtd_output_normalized)
                    break

        rtd_full_probs = np.concatenate(unsorted_outs_rtd_electra, axis=1)
        binary_labels = np.concatenate(binary_labels, axis=1)
        pad_labels = np.concatenate(pad_labels, axis=1)

        if len(rtd_full_probs.shape) == 3:
            rtd_full_probs = np.expand_dims(rtd_full_probs, 0)

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            unsorted_outs = np.concatenate(unsorted_outs, axis=0)  # axis 0 is inference batch
            repr_list = np.concatenate(repr_list, axis=0)

            unsorted_outs = np.reshape(unsorted_outs, (masks_len, batch_size, rmd_classes))
            unsorted_outs = np.transpose(unsorted_outs, (1, 0, 2))

            scores_ne = np.zeros((unsorted_outs.shape[0], unsorted_outs.shape[1]))
            scores_pl = np.zeros((unsorted_outs.shape[0], unsorted_outs.shape[1]))
            scores_mp = np.zeros((unsorted_outs.shape[0], unsorted_outs.shape[1]))

        unsorted_outs_rtd = np.concatenate(unsorted_outs_rtd, axis=0)
        unsorted_outs_rtd = np.reshape(unsorted_outs_rtd, (masks_len, batch_size, self.extra_args["max_seq_length"]))
        unsorted_outs_rtd = np.transpose(unsorted_outs_rtd, (1, 0, 2))

        scores_pl_electra = np.zeros(
            (unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))
        scores_pl_electra_corrupt = np.zeros(
            (unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))
        scores_pl_electra_clean = np.zeros(
            (unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))

        scores_ne_electra_2 = np.zeros((unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))
        scores_pl_electra_2 = np.zeros((unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))
        scores_mp_electra_2 = np.zeros((unsorted_outs_rtd.shape[0], unsorted_outs_rtd.shape[1]))

        self.current_batch_scores_pl = np.full_like(unsorted_outs_rtd, np.inf)

        short_anomalies = 0

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            for b_el in range(unsorted_outs.shape[0]):  #batch elem
                for msk_idx in range(unsorted_outs.shape[1]):
                    scores_ne[b_el][msk_idx] = neg_entropy(unsorted_outs[b_el][msk_idx][:]) #batch_el, msk_id, probs
                    scores_pl[b_el][msk_idx] = unsorted_outs[b_el][msk_idx][msk_idx]
                    scores_mp[b_el][:] = np.max(unsorted_outs[b_el][:][:], axis=1)

                    score_mp_electra_2 = []
                    score_ne_electra_2 = []
                    score_pl_electra_2 = []

                    seq_len = np.count_nonzero(pad_labels[b_el][msk_idx]==0)

                    for seq_el in range(rtd_full_probs.shape[-2]):  # for every el in seq
                        if seq_el == 0 or seq_el == rtd_full_probs.shape[-2] - 1:
                            continue

                        label_idx = binary_labels[b_el][msk_idx][seq_el]

                        score_ne_val = neg_entropy(rtd_full_probs[b_el][msk_idx][seq_el][:])
                        score_pl_val = rtd_full_probs[b_el][msk_idx][seq_el][label_idx]
                        score_mp_val = np.max(rtd_full_probs[b_el][msk_idx][seq_el][:])

                        if "short_anomalies" in self.extra_args:
                            if seq_len <= self.extra_args['short_anomalies']:
                                score_ne_val = 0
                                score_pl_val = 0
                                score_mp_val = 0

                        if pad_labels[b_el][msk_idx][seq_el] == 0:
                            score_mp_electra_2.append(score_mp_val)
                            score_ne_electra_2.append(score_ne_val)
                            self.current_batch = batch.numpy()
                            self.current_batch_scores_pl[b_el][msk_idx][seq_el] = score_pl_val
                            score_pl_electra_2.append(score_pl_val)

                    scores_ne_electra_2[b_el][msk_idx] = np.mean(score_ne_electra_2)
                    scores_pl_electra_2[b_el][msk_idx] = np.mean(score_pl_electra_2)
                    scores_mp_electra_2[b_el][msk_idx] = np.mean(score_mp_electra_2)
                    break

            scores_ne = scores_ne.mean(axis=-1)
            scores_pl = scores_pl.mean(axis=-1)
            scores_mp = scores_mp.mean(axis=-1)

            scores_ne_electra_2 = scores_ne_electra_2.mean(axis=-1)
            scores_pl_electra_2 = scores_pl_electra_2.mean(axis=-1)
            scores_mp_electra_2 = scores_mp_electra_2.mean(axis=-1)

            return scores_pl, scores_mp, scores_ne, scores_pl_electra_2, scores_mp_electra_2, scores_ne_electra_2, repr_list
        else:
            for b_el in range(unsorted_outs_rtd.shape[0]):  #batch elem
                for msk_idx in range(self.extra_args['vanilla_electra']['no_masks']):

                    score_mp_electra_2 = []
                    score_ne_electra_2 = []
                    score_pl_electra_2 = []

                    for seq_el in range(rtd_full_probs.shape[-2]):  # for every el in seq
                        if seq_el == 0 or seq_el == rtd_full_probs.shape[-2] - 1:
                            continue
                        # score_ne_electra[seq_el] = neg_entropy(rtd_full_probs[b_el][msk_idx][seq_el][:])
                        label_idx = binary_labels[b_el][msk_idx][seq_el]
                        # score_pl_electra[seq_el] = rtd_full_probs[b_el][msk_idx][seq_el][label_idx]
                        # score_mp_electra[seq_el] = np.max(rtd_full_probs[b_el][msk_idx][seq_el][:])

                        score_ne_val = neg_entropy(rtd_full_probs[b_el][msk_idx][seq_el][:])
                        score_pl_val = rtd_full_probs[b_el][msk_idx][seq_el][label_idx]
                        score_mp_val = np.max(rtd_full_probs[b_el][msk_idx][seq_el][:])

                        if pad_labels[b_el][msk_idx][seq_el] == 0:
                            score_mp_electra_2.append(score_mp_val)
                            score_ne_electra_2.append(score_ne_val)
                            self.current_batch = batch.numpy()
                            self.current_batch_scores_pl[b_el][msk_idx][seq_el] = score_pl_val
                            score_pl_electra_2.append(score_pl_val)

                    scores_ne_electra_2[b_el][msk_idx] = np.mean(score_ne_electra_2)
                    scores_pl_electra_2[b_el][msk_idx] = np.mean(score_pl_electra_2)
                    scores_mp_electra_2[b_el][msk_idx] = np.mean(score_mp_electra_2)
                    break

            scores_ne_electra_2 = scores_ne_electra_2.mean(axis=-1)
            scores_pl_electra_2 = scores_pl_electra_2.mean(axis=-1)
            scores_mp_electra_2 = scores_mp_electra_2.mean(axis=-1)

            return scores_pl_electra_2, score_mp_electra_2, scores_ne_electra_2

    def train_anomaly(
        self,
        train_dataset,
        eval_dataset,
        eval_doc_dataset,
        eval_doc_dataset_outliers,
        show_running_loss=True,
        verbose=True,
        sched_params=None,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        masks = self.masks

        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        if self.extra_args['random_generator'] == False:
            logger.info(' USING GENERATOR LOSS')

        if self.extra_args['use_rtd_loss'] == True:
            logger.info(' USING RTD LOSS')

        if "vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False:
            logger.info(' USING ELECTRA-VANILLA-OD')

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)

        if self.is_world_master():
            tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
            self.tb_writer = tb_writer

        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      sampler=train_sampler,
                                      collate_fn=collate,
                                      num_workers=4)

        if self.extra_args['extract_repr']:
            train_sampler_repr = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader_repr = DataLoader(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler_repr,
                                        collate_fn=collate,
                                        num_workers=4)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps //
                (len(train_dataloader) // args.gradient_accumulation_steps) +
                1)
        else:
            t_total = len(
                train_dataloader
            ) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                # generator
                "params": [
                    p for n, p in model.generator_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                args.weight_decay,
                "lr":
                args.learning_rate * self.extra_args["mlm_lr_ratio"],
            },
            {
                # generator w/o weight_decay
                "params": [
                    p for n, p in model.generator_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr":
                args.learning_rate * self.extra_args["mlm_lr_ratio"],
            }
        ]

        # discriminator
        disc_unique_params = []
        for n, pgr in model.discriminator_model.named_parameters():
            if n not in [
                    "electra.embeddings.word_embeddings.weight",
                    "electra.embeddings.position_embeddings.weight",
                    "electra.embeddings.token_type_embeddings.weight"
            ]:
                disc_unique_params.append((n, pgr))
                # print("Disc unique", n, pgr.shape)

        optimizer_grouped_parameters.append({
            "params": [
                p for n, p in disc_unique_params
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
            "lr":
            args.learning_rate,
        })

        optimizer_grouped_parameters.append({
            "params": [
                p for n, p in disc_unique_params
                if any(nd in n for nd in no_decay)
            ],
            "lr":
            args.learning_rate,
        })

        # optimizer
        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              eps=args.adam_epsilon,
                              amsgrad=True)
        else:
            optimizer = SGD(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            momentum=0.9,
                            weight_decay=args.weight_decay)
        pytorch_total_params = sum(p.numel() for p in model.parameters()
                                   if p.requires_grad)

        optimizer_total_params = 0
        for pgroup in optimizer_grouped_parameters:
            optimizer_total_params += sum(p.numel() for p in pgroup["params"])

        assert (pytorch_total_params == optimizer_total_params)

        tb_writer.add_text(f'total-params', str(pytorch_total_params), 0)
        logger.info(f' TOTAL PARAMETERS: {pytorch_total_params}')

        tb_writer.add_text(f'tokenizer-len', str(len(self.tokenizer)), 0)
        logger.info(f' TOKENIZER LEN: {len(self.tokenizer)}')

        tb_writer.add_text(f'args', str(self.extra_args))

        if sched_params is not None:
            if sched_params['sched_name'] == 'plateau':
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=sched_params['factor'],
                    patience=sched_params['patience'],
                    verbose=sched_params['verbose'],
                    threshold=sched_params['threshold'],
                    min_lr=sched_params['min_lr'])
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(" Training started")

        global_step = 0
        tr_loss, logging_loss, tr_loss_disc, tr_loss_disc_electra, tr_loss_gen, logging_loss_disc, logging_loss_disc_electra, logging_loss_gen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        tr_acc, logging_acc = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs),
                                desc="Epoch",
                                disable=args.silent,
                                mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                **kwargs)

        model.train()
        preds_total, gt_total = None, None
        for current_epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(current_epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"iteration{global_step}", disable=args.silent)):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                loss = 0

                if "vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False:
                    inputs, labels = mask_tokens_vanilla(batch, tokenizer, args) if args.mlm else (batch, batch)
                else:
                    inputs, labels, labels_clf = mask_tokens(
                        batch, tokenizer, masks, args,
                        train=True) if args.mlm else (batch, batch)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                    labels_clf = labels_clf.to(self.device)

                outputs = model(inputs, masked_lm_labels=labels)

                if "vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False:
                    g_loss = outputs[0]
                    d_electra_loss = outputs[3]
                    sampled_tokens = outputs[2]
                else:
                    g_loss = outputs[0]  # generator
                    d_electra_loss = outputs[4]  # rtd
                    d_output = outputs[2]  # rmd
                    sampled_tokens = outputs[3]

                    labels_clf = labels_clf.squeeze(-1)
                    accuracy, preds, gt = calculate_acc(d_output, labels_clf)
                    if preds_total is None or gt_total is None:
                        preds_total, gt_total = preds, gt
                    else:
                        preds_total = np.append(preds_total, preds)
                        gt_total = np.append(gt_total, gt)

                    d_loss = self.CELoss(d_output, labels_clf.long())

                    loss += (self.extra_args["rmd_loss_weight"] * d_loss)  #RMD Loss

                if self.extra_args['use_rtd_loss'] == True:
                    loss += (self.extra_args["rtd_loss_weight"] *
                             d_electra_loss)  #RTD Loss

                if self.extra_args['random_generator'] == False:
                    if 'train_just_generator' in self.extra_args:
                        if self.extra_args['train_just_generator'] == 0:
                            loss += (self.extra_args['mlm_loss_weight'] * g_loss)  #MLM Loss
                    else:
                        loss += (self.extra_args['mlm_loss_weight'] * g_loss)  #MLM Loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                    if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                        d_loss = d_loss.mean()

                    g_loss = g_loss.mean()

                current_loss = loss.item()
                if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                    current_loss_disc = d_loss.item()
                current_loss_electra_disc = d_electra_loss.item()
                current_loss_gen = g_loss.item()

                if (type(scheduler) ==
                        torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched_lr = optimizer.param_groups[0]['lr']
                else:
                    sched_lr = scheduler.get_lr()[0]

                if show_running_loss:
                    if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                        print(f"\rloss: {loss:.4f}, disc: {current_loss_disc:.4f}, e: {current_loss_electra_disc:.4f}, gen: {current_loss_gen:.4f}, acc: {accuracy:.2f}, lr={sched_lr:.2f}", end="")
                    else:
                        print(f"\rloss: {loss:.4f}, e: {current_loss_electra_disc:.4f}, gen: {current_loss_gen:.4f}, lr={sched_lr:.2f}", end="")

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if 'train_just_generator' in self.extra_args:
                    if global_step >= self.extra_args['train_just_generator']:
                        loss.backward()
                    else:
                        g_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                if not ("vanilla_electra" in self.extra_args
                        and self.extra_args["vanilla_electra"] != False):
                    tr_loss_disc += current_loss_disc
                    tr_acc += accuracy.item()

                tr_loss_disc_electra += current_loss_electra_disc
                tr_loss_gen += current_loss_gen

                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)

                optimizer.step()
                if (type(scheduler) ==
                        torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(tr_loss)
                else:
                    scheduler.step()  # Update learning rate schedule

                model.zero_grad()
                global_step += 1
                self.global_step = global_step

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if "plot_conf_mtx" in self.extra_args:
                        if self.extra_args['plot_conf_mtx']:
                            cm = sklearn.metrics.confusion_matrix(
                                gt_total, preds_total)
                            cm_fig = plot_confusion_matrix(cm,
                                                           class_names=range(
                                                               0, len(gt)))
                            cm_img = plot_to_image(cm_fig)
                            tb_writer.add_image("train_cm", cm_img,
                                                global_step)

                    preds_total, gt_total = None, None

                    if self.is_world_master():
                        tb_writer.add_scalar("train/lr", sched_lr, global_step)
                        tb_writer.add_scalar("train/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                            tb_writer.add_scalar("train/loss_rmd", (tr_loss_disc - logging_loss_disc) / args.logging_steps, global_step)
                        tb_writer.add_scalar("train/loss_rtd", (tr_loss_disc_electra - logging_loss_disc_electra) / args.logging_steps, global_step)
                        tb_writer.add_scalar("train/loss_gen", (tr_loss_gen - logging_loss_gen) / args.logging_steps, global_step)
                        tb_writer.add_scalar("train/acc", (tr_acc - logging_acc) / args.logging_steps, global_step)

                    logging_loss = tr_loss

                    if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                        logging_loss_disc = tr_loss_disc

                    logging_loss_disc_electra = tr_loss_disc_electra
                    logging_loss_gen = tr_loss_gen
                    logging_acc = tr_acc

                if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps
                        == 0):
                    logger.info(' EVALUATING ANOMALIES (NORMAL) [in-epoch]')
                    results = self.eval_model_anomaly(
                        eval_dataset,
                        global_step=global_step,
                        **kwargs,
                    )

                    if self.is_world_master():
                        for key, value in results.items():
                            tb_writer.add_scalar("eval/{}".format(key), value,
                                                 global_step)

                if global_step == 1:
                    logger.info('EVAL AT STEP 0')
                    result_docs = self.eval_model_document(
                            eval_doc_dataset,
                            eval_doc_dataset_outliers,
                            global_step=global_step,
                            **kwargs,
                        )
                    if self.is_world_master():
                        for key, value in result_docs.items():
                            tb_writer.add_scalar("anomaly/{}".format(key),
                                                 value, global_step)


                if args.evaluate_during_training and (args.evaluate_during_training_steps > 0 and global_step >= self.extra_args['eval_anomaly_after'] and global_step % self.extra_args["evaluate_during_training_steps_anomaly"] == 0):

                    logger.info(' EVALUATING ANOMALIES (DOCS) [in-epoch]')
                    if "dump_histogram_epochs" in self.extra_args:
                        if self.current_epoch in self.extra_args["dump_histogram_epochs"]:
                            result_docs = self.eval_model_document(
                                eval_doc_dataset,
                                eval_doc_dataset_outliers,
                                global_step=global_step,
                                **kwargs,
                            )
                            self.extra_args["dump_histogram_epochs"].remove(self.current_epoch)
                    else:
                        result_docs = self.eval_model_document(
                            eval_doc_dataset,
                            eval_doc_dataset_outliers,
                            global_step=global_step,
                            **kwargs,
                        )

                        if self.extra_args['extract_repr']:
                            self.outlier_repr_list
                            self.inlier_repr_list

                            with torch.no_grad():
                                train_reprs = []
                                for batch in tqdm(train_dataloader_repr, desc='EXTRACTING REPR (train)'):
                                    mask_repr = {
                                        'mask' : [False for _ in range(self.extra_args['block_size']-2)],
                                        'label' : 0
                                    }
                                    inputs, labels, _ = mask_tokens(batch, tokenizer, masks, args, custom_mask=mask_repr, train=False, no_mask=True)
                                    inputs = inputs.to(self.device)
                                    labels = labels.to(self.device)
                                    output = model(inputs, masked_lm_labels=labels, replace_tokens=False) if args.mlm else model(inputs, labels=labels)
                                    representations = output[7].cpu()
                                    train_reprs.append(representations)
                                train_reprs = np.concatenate(train_reprs, axis=0)
                                import pickle as pkl
                                with open(f'./representations/{self.extra_args["subset_name"]}_train_step{self.global_step}.pkl', 'wb') as fp:
                                    pkl.dump(train_reprs, fp)
                                with open(f'./representations/{self.extra_args["subset_name"]}_inliers_step{self.global_step}.pkl', 'wb') as fp:
                                    pkl.dump(self.inlier_repr_list, fp)
                                with open(f'./representations/{self.extra_args["subset_name"]}_outliers_step{self.global_step}.pkl', 'wb') as fp:
                                    pkl.dump(self.outlier_repr_list, fp)
                                logger.info(" DUMPED: ")
                                logger.info(f'{self.extra_args["subset_name"]}_train_step{self.global_step}.pkl')
                                logger.info(f'{self.extra_args["subset_name"]}_inliers_step{self.global_step}.pkl')
                                logger.info(f'{self.extra_args["subset_name"]}_outliers_step{self.global_step}.pkl')

                    if self.is_world_master():
                        for key, value in result_docs.items():
                            tb_writer.add_scalar("anomaly/{}".format(key),
                                                 value, global_step)

                if args.max_steps > 0 and global_step > args.max_steps:
                    return global_step, tr_loss / global_step

            epoch_number += 1
            self.current_epoch = epoch_number

            if args.max_steps > 0 and global_step > args.max_steps:
                return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step

    def eval_model_anomaly(self,
                           eval_dataset,
                           global_step=0,
                           tb_writer=None,
                           **kwargs):

        self._move_model_to_device()

        with torch.no_grad():
            result = self.evaluate_anomaly(eval_dataset,
                                           global_step=global_step,
                                           **kwargs)
        self.results.update(result)

        return result

    def calculate_scores(self, dataloader, tqdm_name):
        """ Should be in utils.py
        """
        first = True
        idx_list = dataloader.dataset.idx_l
        self.idx_list = idx_list

        self.dataset_scores_pl_electra = []

        for batch in tqdm(dataloader, desc=tqdm_name):
            if first == True:
                if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                    scores_pl, scores_mp, scores_ne, scores_pl_electra_2, scores_mp_electra_2, scores_ne_electra_2, repr_list = self.test_anomaly(batch.long())
                    self.repr_list = repr_list
                    self.current_data = self.current_batch
                    self.current_scores_pl = self.current_batch_scores_pl
                else:
                    scores_pl_electra_2, scores_mp_electra_2, scores_ne_electra_2 = self.test_anomaly(batch.long())
                    self.current_data = self.current_batch
                    self.current_scores_pl = self.current_batch_scores_pl

                first = False
            else:
                if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                    batch_scores_pl, batch_scores_mp, batch_scores_ne, batch_scores_pl_electra_2, batch_scores_mp_electra_2, batch_scores_ne_electra_2, repr_list = self.test_anomaly(batch.long())
                    scores_ne = np.concatenate((scores_ne, batch_scores_ne), axis=0)
                    scores_pl = np.concatenate((scores_pl, batch_scores_pl), axis=0)
                    scores_mp = np.concatenate((scores_mp, batch_scores_mp), axis=0)
                    self.repr_list = np.concatenate((self.repr_list, repr_list), axis=0)
                    self.current_data = np.concatenate((self.current_data, self.current_batch), axis=0)
                    self.current_scores_pl = np.concatenate((self.current_scores_pl, self.current_batch_scores_pl), axis=0)
                else:
                    batch_scores_pl_electra_2, batch_scores_mp_electra_2, batch_scores_ne_electra_2 = self.test_anomaly(batch.long())
                    self.current_data = np.concatenate((self.current_data, self.current_batch), axis=0)
                    self.current_scores_pl = np.concatenate((self.current_scores_pl, self.current_batch_scores_pl), axis=0)

                scores_ne_electra_2 = np.concatenate((scores_ne_electra_2, batch_scores_ne_electra_2), axis=0)

                scores_pl_electra_2 = np.concatenate((scores_pl_electra_2, batch_scores_pl_electra_2), axis=0)
                scores_mp_electra_2 = np.concatenate((scores_mp_electra_2, batch_scores_mp_electra_2), axis=0)

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            scores_ne = np.split(scores_ne, idx_list)
            scores_ne = np.array([np.mean(score) for score in scores_ne])

            scores_pl = np.split(scores_pl, idx_list)
            scores_pl = np.array([np.mean(score) for score in scores_pl])

            scores_mp = np.split(scores_mp, idx_list)
            scores_mp = np.array([np.mean(score) for score in scores_mp])

        self.current_data = np.split(self.current_data, idx_list)
        self.current_scores_pl = np.split(self.current_scores_pl, idx_list)
        self.repr_list = np.split(self.repr_list, idx_list)

        scores_pl_electra_2 = np.split(scores_pl_electra_2, idx_list)
        scores_pl_electra_2 = np.array([np.mean(score) for score in scores_pl_electra_2])

        scores_mp_electra_2 = np.split(scores_mp_electra_2, idx_list)
        scores_mp_electra_2 = np.array([np.mean(score) for score in scores_mp_electra_2])

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            return scores_pl, scores_mp, scores_ne, scores_pl_electra_2, scores_mp_electra_2, scores_ne_electra_2
        else:
            return scores_pl_electra_2, scores_mp_electra_2, scores_ne_electra_2

    def get_text_samples(self, anomaly_results): #inlier, then anomaly
        for name in anomaly_results:
            cutoff = anomaly_results[name]['cutoff']

            anomaly_results[name]['pred_inlier_idx'] = np.argwhere(anomaly_results[name]['pred_scores'] > cutoff)  # idx-all
            anomaly_results[name]['pred_outlier_idx'] = np.argwhere(anomaly_results[name]['pred_scores'] <= cutoff)  # idx-all

            anomaly_results[name]['fp_idx'] = anomaly_results[name]['pred_inlier_idx'][anomaly_results[name]['pred_inlier_idx'] > self.inlier_len]  #idx-all
            anomaly_results[name]['tp_idx'] = anomaly_results[name]['pred_inlier_idx'][anomaly_results[name]['pred_inlier_idx'] <= self.inlier_len]  #idx-all
            anomaly_results[name]['fn_idx'] = anomaly_results[name]['pred_outlier_idx'][anomaly_results[name]['pred_outlier_idx'] <= self.inlier_len]  #idx-all
            anomaly_results[name]['tn_idx'] = anomaly_results[name]['pred_outlier_idx'][anomaly_results[name]['pred_outlier_idx'] > self.inlier_len]  #idx-all

            anomaly_results[name]['fp_scores'] = anomaly_results[name]['pred_scores'][anomaly_results[name]['fp_idx']]  # values-all
            anomaly_results[name]['fn_scores'] = anomaly_results[name]['pred_scores'][anomaly_results[name]['fn_idx']]  # values-all
            anomaly_results[name]['tp_scores'] = anomaly_results[name]['pred_scores'][anomaly_results[name]['tp_idx']]  # values-all
            anomaly_results[name]['tn_scores'] = anomaly_results[name]['pred_scores'][anomaly_results[name]['tn_idx']]  # values-all

            anomaly_results[name]['fp_top5'] = anomaly_results[name]['fp_scores'].argsort()[-5:][::-1]  # idx-local
            anomaly_results[name]['fn_top5'] = anomaly_results[name]['fn_scores'].argsort()[-5:][::-1]  # idx-local
            anomaly_results[name]['tp_top5'] = anomaly_results[name]['tp_scores'].argsort()[-5:][::-1]  # idx-local
            anomaly_results[name]['tn_top5'] = anomaly_results[name]['tn_scores'].argsort()[-5:][::-1]  # idx-local

            tokenizer = self.tokenizer

            all_tokens_scores_pl = []
            all_tokens_scores_pl.extend(self.inlier_current_scores_pl)
            all_tokens_scores_pl.extend(self.outlier_current_scores_pl)

            def get_wordpiece_tokens(raw_text, tokenizer):
                tokens = tokenizer.encode(raw_text)
                tokens_normal = tokenizer.decode(tokens).split(' ')
                text_wordpiece = ''
                for token in tokens:
                    text_wordpiece += f'{tokenizer._convert_id_to_token(token)} '
                tokens_wordpiece = text_wordpiece.split(' ')
                return tokens_wordpiece, tokens_normal

            def extract_scores(tokens, tokens_normal, scores, export_path, global_step, name):
                text_df = np.array(tokens)
                text_df = np.expand_dims(text_df, 0)
                text_df = pd.DataFrame(text_df)

                text_df_normal = np.array(tokens_normal)
                text_df_normal = np.expand_dims(text_df_normal, 0)
                text_df_normal = pd.DataFrame(text_df_normal)

                scores_df = pd.DataFrame(scores)
                result_df = pd.concat([text_df_normal, text_df], ignore_index=True, sort=False)
                result_df = pd.concat([result_df, scores_df], ignore_index=True, sort=False)

                full_path = os.path.join(export_path, name)

                if not os.path.exists(full_path):
                    os.makedirs(full_path)

                # result_df.to_csv(f'{full_path}/step_{global_step}.csv')
                # logger.info(f' Succesfully extracted {name}')

            for k, idx in enumerate(anomaly_results[name]['fp_top5']):
                raw_text = self.lines[anomaly_results[name]['fp_idx'][idx]]
                text = raw_text + f"_score_{anomaly_results[name]['fp_scores'][idx]}_cutoff_{cutoff}_idx{idx}_allidx{anomaly_results[name]['fp_idx'][idx]}"

                self.tb_writer.add_text(f'{name}/fp_top5-{k}', text, self.global_step)

                sample_idx = anomaly_results[name]['fp_idx'][idx]
                scores = all_tokens_scores_pl[sample_idx]
                scores = scores.reshape((scores.shape[1], scores.shape[0]*scores.shape[2]))

                tokens_wordpiece, tokens_normal = get_wordpiece_tokens(raw_text, tokenizer)
                if self.extra_args['extract_scores'] != 0:
                    extract_scores(tokens_wordpiece, tokens_normal, scores, self.extra_args['scores_export_path'], self.global_step, f'fp{k}')

            for k, idx in enumerate(anomaly_results[name]['fn_top5']):
                raw_text = self.lines[anomaly_results[name]['fn_idx'][idx]]
                text = raw_text + f"_score_{anomaly_results[name]['fn_scores'][idx]}_cutoff_{cutoff}_idx{idx}_allidx{anomaly_results[name]['fn_idx'][idx]}"

                self.tb_writer.add_text(f'{name}/fn_top5-{k}', text, self.global_step)

                sample_idx = anomaly_results[name]['fn_idx'][idx]
                scores = all_tokens_scores_pl[sample_idx]
                scores = scores.reshape((scores.shape[1], scores.shape[0]*scores.shape[2]))

                tokens_wordpiece, tokens_normal = get_wordpiece_tokens(raw_text, tokenizer)
                if self.extra_args['extract_scores'] != 0:
                    extract_scores(tokens_wordpiece, tokens_normal, scores, self.extra_args['scores_export_path'], self.global_step, f'fn{k}')


            for k, idx in enumerate(anomaly_results[name]['tp_top5']):
                raw_text = self.lines[anomaly_results[name]['tp_idx'][idx]]
                text = raw_text + f"_score_{anomaly_results[name]['tp_scores'][idx]}_cutoff_{cutoff}_idx{idx}_allidx{anomaly_results[name]['tp_idx'][idx]}"
                self.tb_writer.add_text(f'{name}/tp_top5-{k}', text, self.global_step)

                sample_idx = anomaly_results[name]['tp_idx'][idx]
                scores = all_tokens_scores_pl[sample_idx]
                scores = scores.reshape((scores.shape[1], scores.shape[0]*scores.shape[2]))

                tokens_wordpiece, tokens_normal = get_wordpiece_tokens(raw_text, tokenizer)
                if self.extra_args['extract_scores'] != 0:
                    extract_scores(tokens_wordpiece, tokens_normal, scores, self.extra_args['scores_export_path'], self.global_step, f'tp{k}')


            for k, idx in enumerate(anomaly_results[name]['tn_top5']):
                raw_text = self.lines[anomaly_results[name]['tn_idx'][idx]]
                text = raw_text + f"_score_{anomaly_results[name]['tn_scores'][idx]}_cutoff_{cutoff}_idx{idx}_allidx{anomaly_results[name]['tn_idx'][idx]}"
                self.tb_writer.add_text(f'{name}/tn_top5-{k}', text, self.global_step)

                sample_idx = anomaly_results[name]['tn_idx'][idx]
                scores = all_tokens_scores_pl[sample_idx]
                scores = scores.reshape((scores.shape[1], scores.shape[0]*scores.shape[2]))

                tokens_wordpiece, tokens_normal = get_wordpiece_tokens(raw_text, tokenizer)
                if self.extra_args['extract_scores'] != 0:
                    extract_scores(tokens_wordpiece, tokens_normal, scores, self.extra_args['scores_export_path'], self.global_step, f'tn{k}')

    def evaluate_anomaly_docs(self,
                              eval_dataset,
                              eval_outlier_dataset,
                              global_step=0,
                              **kwargs):
        """
        Evaluates the model on eval_dataset.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        masks = self.masks
        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        normal_scores = None
        anomaly_scores = None

        results = {}

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)

        eval_bs = self.extra_args["anomaly_batch_size"]
        num_workers = 4

        eval_outlier_sampler = SequentialSampler(eval_outlier_dataset)
        eval_outlier_dataloader = DataLoader(eval_outlier_dataset,
                                             sampler=eval_outlier_sampler,
                                             batch_size=eval_bs,
                                             collate_fn=collate,
                                             num_workers=num_workers)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=eval_bs,
            collate_fn=collate,
            num_workers=num_workers,
        )

        logger.info(
            f' LENS outliers/inliers {len(eval_outlier_dataloader.dataset)}, {len(eval_dataloader.dataset)}'
        )
        logger.info(' Calculating outlier scores [doc]')

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        # Should be moved outside this function

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            anomaly_scores_pl, anomaly_scores_mp, anomaly_scores_ne, anomaly_scores_pl_electra_2, anomaly_scores_mp_electra_2, anomaly_scores_ne_electra_2 = self.calculate_scores(eval_outlier_dataloader, 'outlier_sc_doc')

            self.outlier_current_scores_pl = self.current_scores_pl.copy()
            self.outlier_current_data = self.current_data.copy()
            self.outlier_repr_list = np.array([np.mean(x, axis=0) for x in self.repr_list])

            inlier_scores_pl, inlier_scores_mp, inlier_scores_ne, inlier_scores_pl_electra_2, inlier_scores_mp_electra_2, inlier_scores_ne_electra_2 = self.calculate_scores(eval_dataloader, 'inlier_sc_doc')
            self.inlier_current_scores_pl = self.current_scores_pl.copy()
            self.inlier_current_data = self.current_data.copy()
            self.inlier_repr_list = np.array([np.mean(x, axis=0) for x in self.repr_list.copy()])
        else:
            anomaly_scores_pl_electra_2, anomaly_scores_mp_electra_2, anomaly_scores_ne_electra_2 = self.calculate_scores(eval_outlier_dataloader, 'outlier_sc_doc')
            inlier_scores_pl_electra_2, inlier_scores_mp_electra_2, inlier_scores_ne_electra_2 = self.calculate_scores(eval_dataloader, 'inlier_sc_doc')

        logger.info(f" Final shape [outlier] {anomaly_scores_pl_electra_2.shape}")
        logger.info(f" Final shape [inlier ] {inlier_scores_pl_electra_2.shape}")

        if self.extra_args['eval_anomalies']:

            if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
                real_preds_ne, cutoff_ne = self.get_preds_calc_metrics(
                    inlier_scores_ne,
                    anomaly_scores_ne,
                    calc_dummy=False,
                    name='NE',
                    global_step=global_step)

                real_preds_pl, cutoff_pl = self.get_preds_calc_metrics(
                    inlier_scores_pl,
                    anomaly_scores_pl,
                    calc_dummy=False,
                    name='PL',
                    global_step=global_step)

                real_preds_mp, cutoff_mp = self.get_preds_calc_metrics(
                    inlier_scores_mp,
                    anomaly_scores_mp,
                    calc_dummy=False, name='MP',
                    global_step=global_step)

                results["NE/inlier_mean"] = inlier_scores_ne.mean()
                results["MP/inlier_mean"] = inlier_scores_mp.mean()
                results["PL/inlier_mean"] = inlier_scores_pl.mean()

                results["NE/outlier_mean"] = anomaly_scores_ne.mean()
                results["MP/outlier_mean"] = anomaly_scores_mp.mean()
                results["PL/outlier_mean"] = anomaly_scores_pl.mean()

                results["NE/mean"] = (results["NE/inlier_mean"] + results["NE/outlier_mean"]) / 2
                results["MP/mean"] = (results["MP/inlier_mean"] + results["MP/outlier_mean"]) / 2
                results["PL/mean"] = (results["PL/inlier_mean"] +
                                      results["PL/outlier_mean"]) / 2

                results["NE/roc_auc"], results["NE/pr_auc_in"], results["NE/pr_auc_out"] = real_preds_ne
                results["PL/roc_auc"], results["PL/pr_auc_in"], results["PL/pr_auc_out"] = real_preds_pl
                results["MP/roc_auc"], results["MP/pr_auc_in"], results["MP/pr_auc_out"] = real_preds_mp

            real_preds_ne_electra_bug, cutoff_ne_electra_bug = self.get_preds_calc_metrics(
                inlier_scores_ne_electra_2,
                anomaly_scores_ne_electra_2,
                calc_dummy=False,
                name='NE_RTD',
                global_step=global_step)

            real_preds_pl_electra_bug, cutoff_pl_electra_bug = self.get_preds_calc_metrics(
                inlier_scores_pl_electra_2,
                anomaly_scores_pl_electra_2,
                calc_dummy=False,
                name='PL_RTD',
                global_step=global_step)
            real_preds_mp_electra_bug, cutoff_mp_electra_bug = self.get_preds_calc_metrics(
                inlier_scores_mp_electra_2,
                anomaly_scores_mp_electra_2,
                calc_dummy=False,
                name='MP_RTD',
                global_step=global_step)

            results["NE_RTD/roc_auc"], results["NE_RTD/pr_auc_in"], results["NE_RTD/pr_auc_out"] = real_preds_ne_electra_bug
            results["PL_RTD/roc_auc"], results["PL_RTD/pr_auc_in"], results["PL_RTD/pr_auc_out"] = real_preds_pl_electra_bug
            results["MP_RTD/roc_auc"], results["MP_RTD/pr_auc_in"], results["MP_RTD/pr_auc_out"] = real_preds_mp_electra_bug

            anomaly_results_cutoff = {
                "PL_RTD": {
                    "cutoff":
                    cutoff_pl_electra_bug,
                    "pred_scores":
                    np.concatenate((inlier_scores_pl_electra_2, anomaly_scores_pl_electra_2))
                },

            }

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            self.get_text_samples(anomaly_results_cutoff) #inlier, then anomaly

        return results

    def eval_model_document(self,
                            eval_doc_dataset,
                            eval_doc_dataset_outliers,
                            global_step=0,
                            **kwargs):
        """
        Evaluates the model on eval_df. Saves results to args['output_dir']
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        self._move_model_to_device()

        tokenizer = self.tokenizer
        args = self.args
        special_tokens_count = 2

        with torch.no_grad():
            result = self.evaluate_anomaly_docs(eval_doc_dataset,
                                                eval_doc_dataset_outliers,
                                                global_step=global_step,
                                                **kwargs)

        self.results.update(result)

        return result

    def get_preds_calc_metrics(self,
                               normal_scores,
                               anomaly_scores,
                               calc_dummy=True,
                               name='NE',
                               global_step=0):

        logger.info(f' CALCULATING ANOMALY METRICS ({name})')

        labels_normal = np.ones_like(normal_scores)
        labels_anomaly = np.zeros_like(anomaly_scores)
        gt = np.concatenate((labels_normal, labels_anomaly))
        preds = np.concatenate((normal_scores, anomaly_scores))

        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)

        if self.extra_args['dump_histogram'] != 0:
            if global_step % self.extra_args['dump_histogram'] == 0:# and global_step != 0:
                labels_and_scores = {
                    "ground_truth": gt,
                    "anomaly_scores": preds,
                }

                with open(
                        f'./histogram/{self.extra_args["subset_name"]}_step{global_step}_e{self.current_epoch}_{name}.pkl',
                        'wb') as fp:
                    pkl.dump(labels_and_scores, fp)
                logger.info(f'!!!Succesfully dumped the anomaly scores {name}!!!')
                del labels_and_scores

        roc_auc, pr_auc_norm, pr_auc_anom, cutoff = get_metrics(gt, preds)
        real_preds = (roc_auc, pr_auc_norm, pr_auc_anom)

        print(name)
        print(f'ROC-AUC    : {roc_auc:4f}')
        print(f'PR-AUC-in  : {pr_auc_norm:4f}')
        print(f'PR-AUC-out : {pr_auc_anom:4f}')
        compute_all_scores(anomaly_scores, normal_scores, "./")

        if calc_dummy:
            return real_preds, freq_preds, strat_preds, unif_preds
        else:
            return real_preds, cutoff

    def evaluate_anomaly(self, eval_dataset, global_step=0, **kwargs):
        """
        Evaluates the model on eval_dataset.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        if not ("vanilla_electra" in self.extra_args
                and self.extra_args["vanilla_electra"] != False):
            masks = self.masks

        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        results = {}

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size,
                                     collate_fn=collate,
                                     num_workers=4)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        eval_loss = 0.0
        eval_rtd_loss = 0.0
        if not ("vanilla_electra" in self.extra_args
                and self.extra_args["vanilla_electra"] != False):
            eval_rmd_loss = 0.0
        eval_gen_loss = 0.0
        nb_eval_steps = 0

        accs = []

        for idx, batch in enumerate(tqdm(eval_dataloader, desc="inlier_sc")):
            loss = 0
            initial_text_inlier = batch
            if not ("vanilla_electra" in self.extra_args
                    and self.extra_args["vanilla_electra"] != False):
                inputs, labels, labels_clf = mask_tokens(
                    batch, tokenizer, masks, args,
                    train=False) if args.mlm else (batch, batch)
                labels_clf = labels_clf.to(self.device)
            else:
                inputs, labels = mask_tokens_vanilla(
                    batch, tokenizer, args) if args.mlm else (batch, batch)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)


            outputs = model(inputs, masked_lm_labels=labels)

            if "vanilla_electra" in self.extra_args and self.extra_args[
                    "vanilla_electra"] != False:
                g_loss = outputs[0]
                d_electra_loss = outputs[3]
                sampled_tokens = outputs[2]
            else:
                g_loss = outputs[0]
                d_electra_loss = outputs[4]
                d_output = outputs[2]
                sampled_tokens = outputs[3]

                labels_clf = labels_clf.squeeze(-1)
                accuracy, preds, gt = calculate_acc(d_output, labels_clf)

                accs.append(accuracy.item())

                d_loss = self.CELoss(d_output, labels_clf.long())
                loss += (self.extra_args["rmd_loss_weight"] * d_loss
                         )  #RMD Loss

            if self.extra_args['use_rtd_loss'] == True:
                loss += (self.extra_args["rtd_loss_weight"] * d_electra_loss
                         )  #RTD Loss

            if self.extra_args['random_generator'] == False:
                loss += (self.extra_args['mlm_loss_weight'] * g_loss
                         )  #MLM Loss

            eval_loss += loss.mean().item()
            if not ("vanilla_electra" in self.extra_args
                    and self.extra_args["vanilla_electra"] != False):
                eval_rmd_loss += d_loss.mean().item()
            eval_rtd_loss += d_electra_loss.mean().item()
            eval_gen_loss += g_loss.mean().item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        if not ("vanilla_electra" in self.extra_args
                and self.extra_args["vanilla_electra"] != False):
            eval_rmd_loss = eval_rmd_loss / nb_eval_steps
        eval_rtd_loss = eval_rtd_loss / nb_eval_steps
        eval_gen_loss = eval_gen_loss / nb_eval_steps

        results["loss"] = eval_loss
        results["loss_rtd"] = eval_rtd_loss
        if not ("vanilla_electra" in self.extra_args
                and self.extra_args["vanilla_electra"] != False):
            results["loss_rmd"] = eval_rmd_loss
            results["acc"] = np.mean(accs)

        results["loss_gen"] = eval_gen_loss

        return results

    # @profile
    def load_and_cache_examples_anomaly(self,
                                        file_path,
                                        evaluate=False,
                                        no_cache=False,
                                        verbose=True,
                                        silent=False):
        """
        SHOULD GET DEPRECATED FOR EVALUATION
        Reads a text file from file_path and creates training features.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"

        special_tokens_count = 2
        return SimpleDataset(
            tokenizer,
            self.args,
            file_path,
            mode,
            args.block_size,
            special_tokens_count,
            sliding_window=args.sliding_window,
        )

    def eval_model(self,
                   eval_file,
                   output_dir=None,
                   verbose=True,
                   silent=False,
                   **kwargs):
        """
        Evaluates the model on eval_df. Saves results to args.output_dir
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_file,
                                                    evaluate=True,
                                                    verbose=verbose,
                                                    silent=silent)
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(eval_dataset,
                               output_dir,
                               verbose=verbose,
                               silent=silent,
                               **kwargs)
        self.results.update(result)

        return result

    def evaluate(self,
                 eval_dataset,
                 output_dir,
                 multi_label=False,
                 prefix="",
                 verbose=True,
                 silent=False,
                 **kwargs):
        """
        Evaluates the model on eval_dataset.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        args.silent = True
        eval_output_dir = output_dir
        tokenizer = self.tokenizer

        results = {}

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size,
                                     collate_fn=collate,
                                     num_workers=4)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader,
                          disable=args.silent or silent,
                          desc="Running Evaluation"):
            inputs, labels = mask_tokens(batch, tokenizer, args,
                                         train=False) if args.mlm else (batch,
                                                                        batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = model(
                    inputs, masked_lm_labels=labels) if args.mlm else model(
                        inputs, labels=labels)
                if args.model_type == "electra":
                    g_loss = outputs[0]
                    d_loss = outputs[1]
                    lm_loss = g_loss + args.discriminator_loss_weight * d_loss
                else:
                    lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        results["eval_loss"] = eval_loss
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def load_and_cache_examples(self,
                                file_path,
                                evaluate=False,
                                no_cache=False,
                                verbose=True,
                                silent=False):
        """
        Reads a text file from file_path and creates training features.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, file_path, mode,
                                 args.block_size)
        else:
            dataset_type = args.dataset_type
            if dataset_type == "text":
                return TextDataset(tokenizer,
                                   file_path,
                                   args.block_size,
                                   overwrite_cache=True)
            elif dataset_type == "line_by_line":
                return LineByLineTextDataset(tokenizer, file_path,
                                             args.block_size)
            else:
                special_tokens_count = 3 if bool(
                    args.model_type in ["roberta", "camembert", "xlmroberta"
                                        ]) else 2
                if self.args.max_seq_length > 509 and self.args.model_type != "longformer":
                    self.args.max_seq_length = (509 if bool(
                        args.model_type in
                        ["roberta", "camembert", "xlmroberta"]) else 510)
                    self.args.block_size = (509 if bool(
                        args.model_type in
                        ["roberta", "camembert", "xlmroberta"]) else 510)
                return SimpleDataset(
                    tokenizer,
                    self.args,
                    file_path,
                    mode,
                    args.block_size,
                    special_tokens_count,
                    sliding_window=args.sliding_window,
                )

    def train_tokenizer(self,
                        train_files,
                        tokenizer_name=None,
                        output_dir=None,
                        use_trained_tokenizer=True):
        """
        Train a new tokenizer on `train_files`.
        Args:
        - train_files: List of files to be used when training the tokenizer.
        - tokenizer_name: Name of a pretrained tokenizer or a path to a directory containing a tokenizer.
        - output_dir (optional): The directory where model files will be saved. If not given, self.args.output_dir
        will be used.
        - use_trained_tokenizer (optional): Load the trained tokenizer once training completes.
        Returns: None
        """

        if not self.args.vocab_size:
            raise AttributeError(
                "Cannot train a new tokenizer as vocab_size is not specified in args dict. "
                "Either provide a tokenizer or specify vocab_size.")

        if not isinstance(train_files, list):
            train_files = [train_files]

        if not output_dir:
            output_dir = self.args.output_dir

        if self.args.model_type in ["bert", "electra"]:
            tokenizer = BertWordPieceTokenizer(
                clean_text=self.args.clean_text,
                handle_chinese_chars=self.args.handle_chinese_chars,
                strip_accents=self.args.strip_accents,
                lowercase=self.args.do_lower_case,
            )
            self.args.special_tokens = [
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
            ]
            self.args.wordpieces_prefix = "##"

            tokenizer.train(
                files=train_files,
                vocab_size=self.args.vocab_size,
                min_frequency=self.args.min_frequency,
                special_tokens=self.args.special_tokens,
                wordpieces_prefix="##",
            )
        else:
            tokenizer = ByteLevelBPETokenizer(
                lowercase=self.args.do_lower_case)

            tokenizer.train(
                files=train_files,
                vocab_size=self.args.vocab_size,
                min_frequency=self.args.min_frequency,
                special_tokens=self.args.special_tokens,
            )

        os.makedirs(output_dir, exist_ok=True)

        tokenizer.save_model(output_dir)
        logger.info(" Training of {} tokenizer complete. Saved to {}.".format(
            tokenizer_name, output_dir))

        _, _, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        tokenizer = tokenizer_class.from_pretrained(output_dir)

        if use_trained_tokenizer:
            self.tokenizer = tokenizer
            self.args.tokenizer_name = output_dir
            try:
                if self.args.model_type == "electra":
                    model_to_resize = (self.model.generator_model.module
                                       if hasattr(self.model.generator_model,
                                                  "module") else
                                       self.model.generator_model)
                    model_to_resize.resize_token_embeddings(len(
                        self.tokenizer))

                    model_to_resize = (self.model.discriminator_model.module if
                                       hasattr(self.model.discriminator_model,
                                               "module") else
                                       self.model.discriminator_model)
                    model_to_resize.resize_token_embeddings(len(
                        self.tokenizer))

                model_to_resize = self.model.module if hasattr(
                    self.model, "module") else self.model
                model_to_resize.resize_token_embeddings(len(self.tokenizer))
            except AttributeError:
                pass

    def save_discriminator(self, output_dir=None):
        if self.args.model_type == "electra":
            if not self.args.no_save:
                if not output_dir:
                    output_dir = os.path.join(self.args.output_dir,
                                              "discriminator_model")
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (self.model.discriminator_model.module
                                 if hasattr(self.model.discriminator_model,
                                            "module") else
                                 self.model.discriminator_model)
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError(
                "Model must be of ElectraForLanguageModelingModel type")

    def save_generator(self, output_dir=None):
        if self.args.model_type == "electra":
            if not self.args.no_save:
                if not output_dir:
                    output_dir = os.path.join(self.args.output_dir,
                                              "generator_model")
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (self.model.generator_model.module if hasattr(
                    self.model.generator_model, "module") else
                                 self.model.generator_model)
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError(
                "Model must be of ElectraForLanguageModelingModel type")

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "perplexity": [],
            "eval_loss": [],
            "train_loss": [],
            "acc": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _save_model(self,
                    output_dir=None,
                    optimizer=None,
                    scheduler=None,
                    model=None,
                    results=None):
        if not self.is_world_master():
            return
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            if self.args.model_type in "electra":
                os.makedirs(os.path.join(output_dir, "generator_config"),
                            exist_ok=True)
                os.makedirs(os.path.join(output_dir, "discriminator_config"),
                            exist_ok=True)
                self.generator_config.save_pretrained(
                    os.path.join(output_dir, "generator_config"))
                self.discriminator_config.save_pretrained(
                    os.path.join(output_dir, "discriminator_config"))
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir,
                                               "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(),
                           os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(),
                           os.path.join(output_dir, "scheduler.pt"))
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LanguageModelingArgs()
        args.load(input_dir)
        return args

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]