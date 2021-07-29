import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math


class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, hidden, logits, bias, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()


class Plain(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, labels):
        return F.cross_entropy(logits, labels)



class LabelSmoothing(ClfDistillLossFunction):
    def __init__(self, num_class):
        super(LabelSmoothing, self).__init__()
        self.num_class = num_class

    def forward(self, hidden, logits, bias, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        alphas = (one_hot_labels * torch.exp(bias)).sum(1).unsqueeze(1).expand_as(one_hot_labels)
        target_probs = (1 - alphas) * one_hot_labels + alphas / self.num_class

        example_loss = -(target_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class ReweightBaseline(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        # default we use cuda ....
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = 1 - (one_hot_labels * torch.exp(bias)).sum(1)

        return (weights * loss).sum() / weights.sum()


class BiasProductBaseline(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)
        return F.cross_entropy(logits + bias.float(), labels)