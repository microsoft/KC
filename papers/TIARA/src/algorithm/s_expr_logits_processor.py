# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.generation_logits_process import LogitsProcessor

from algorithm.grailqa_s_expression_checker import GrailQASExpressionChecker
from algorithm.webqsp_s_expression_checker import WebQSPSExpressionChecker


class SExpressionLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        if dataset == 'grail_qa':
            self.checker = GrailQASExpressionChecker()
        elif dataset == 'webqsp':
            self.checker = WebQSPSExpressionChecker()
        self.neg_inf = -10000.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids_cpu = input_ids.cpu()
        scores = scores.cpu().numpy()
        for i in range(0, input_ids_cpu.size(0)):  # for each sequence in beams
            self.sequence_logits_process(i, input_ids_cpu, scores)
        return torch.Tensor(scores).cuda()

    def sequence_logits_process(self, i, input_ids_cpu, scores):
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_cpu[i])
        valid_tokens = self.checker.valid_tokens(input_tokens)
        if valid_tokens is not None:
            scores[i][:] += self.neg_inf
            valid_ids = self.tokenizer.convert_tokens_to_ids(valid_tokens)
            scores[i][valid_ids] -= self.neg_inf
        elif valid_tokens == 'invalid_sequence':
            scores[i][:] += self.neg_inf
