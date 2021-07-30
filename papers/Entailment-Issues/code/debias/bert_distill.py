'''
The code is adapted from https://github.com/UKPLab/emnlp2020-debiasing-unknown/blob/main/src/bert_distill.py

License: Apache License 2.0
'''

from transformers.modeling_bert import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss

from clf_distill_loss_functions import ClfDistillLossFunction


class BertDistill(BertPreTrainedModel):
    """Pre-trained BERT model that uses our loss functions"""
    
    def __init__(self, config, loss_fn):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fn = loss_fn
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None):
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None:
            return logits
        loss = self.loss_fn.forward(pooled_output, logits, bias, labels)
        return logits, loss

    def forward_and_log(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None):
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None:
            return logits
        loss = self.loss_fn.forward(pooled_output, logits, bias, labels)

        cel_fct = CrossEntropyLoss(reduction="none")
        indv_losses = cel_fct(logits, labels).detach()
        return logits, loss, indv_losses
    
    def forward_analyze(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None):
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None and not self.training:
            return logits, pooled_output
        else:
            raise Exception("should be called during eval and "
                            "labels should be none")
