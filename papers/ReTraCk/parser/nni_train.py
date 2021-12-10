# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys
from typing import Dict, Any
import hashlib
from allennlp.commands import main
from allennlp.training.trainer import EpochCallback, GradientDescentTrainer

VALIDATION_METRIC = "avg_f1"


@EpochCallback.register("report_epoch_nni")
class ReportEpochCallBack(EpochCallback):
    def __call__(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        metric_key = "validation_" + VALIDATION_METRIC
        if metric_key in metrics:
            report_metric = metrics[metric_key]
        else:
            report_metric = 0.0
        """@nni.report_intermediate_result(report_metric)"""


def take_hyper_parameters() -> Dict:
    '''@nni.variable(nni.choice(1, 2, 3, 4, 5), name=batch_size)'''
    batch_size = 3

    '''@nni.variable(nni.choice(0.0001, 0.001, 0.0005, 0.00005), name=learning_rate)'''
    learning_rate = 1e-3

    '''@nni.variable(nni.choice(0.00001, 0.00002, 0.00005, 0.000001, 0.000002, 0.000005), name=bert_learning_rate)'''
    bert_learning_rate = 1e-5

    '''@nni.variable(nni.choice("alphabet", "shuffle"), name=entity_order_method)'''
    entity_order_method = "alphabet"

    '''@nni.variable(nni.choice("first", "max", "mean", "shuffle"), name=utterance_agg_method)'''
    utterance_agg_method = "first"

    '''@nni.variable(nni.choice(0.1, 0.2, 0.3, 0.4, 0.5), name=dropout_rate)'''
    dropout_rate = 0.5

    '''@nni.variable(nni.choice(50, 100, 150, 200, 250), name=encoder_hidden)'''
    encoder_hidden = 200

    hyper_parameter_dict = {
        "trainer.num_gradient_accumulation_steps": batch_size,
        "trainer.optimizer.lr": learning_rate,
        "trainer.optimizer.parameter_groups": [
            [
                [
                    ".*text_embedder.*"
                ],
                {
                    "lr": bert_learning_rate
                }
            ]
        ],
        "model.entity_order_method": entity_order_method,
        "model.utterance_agg_method": utterance_agg_method,
        "model.dropout_rate": dropout_rate,
        "model.text_encoder.hidden_size": encoder_hidden,
    }

    return hyper_parameter_dict


if __name__ == '__main__':
    # random a serialization directory
    config_file = "configs/webqsp_bert.jsonnet"

    config = {
        "trainer.checkpointer.num_serialized_models_to_keep": 0,
        "trainer.patience": 10,
        "train_data_path": "D:/users/v-qianl/GrailQA/dataset/webqsp_train_v10_oracle_top5.fix.json",
        "validation_data_path": "D:/users/v-qianl/GrailQA/dataset/webqsp_dev_v10_top5.fix.json",
        "model.fb_roles_file": "D:/users/v-qianl/GrailQA/evaluate/fb_roles.txt",
        "model.fb_types_file": "D:/users/v-qianl/GrailQA/evaluate/fb_types.txt",
        "model.reverse_properties_file": "D:/users/v-qianl/GrailQA/evaluate/reverse_properties.txt",
        "model.enable_f1_eval": True,
        "trainer.epoch_callbacks": [{"type": "report_epoch_nni"}],
        **take_hyper_parameters()
    }

    overrides = json.dumps(config)
    serialization_dir = "D:/users/v-qianl/webqsp_nni/{}".format(hashlib.md5(overrides.encode()).hexdigest())

    # in debug mode.
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "-f",
        "--include-package", "data_reader",
        "--include-package", "grail_parser",
        "-o", overrides
    ]

    main()
