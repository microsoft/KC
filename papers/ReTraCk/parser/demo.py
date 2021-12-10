# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from server import main
import os
from retriever.configs import config_utils

if __name__ == '__main__':

    # Read base environment variables
    data_path = os.environ.get("DATA_PATH")

    if data_path is None:
        print("Warning: DATA_PATH environment variable not set.")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, default="../retrack/configs/demo_grailqa.json")
    cli_args = arg_parser.parse_args()

    print('Using config: ' + cli_args.config_path)

    # Load config
    config = config_utils.get_config(cli_args.config_path)

    model_file = os.path.join(data_path, config["model_path"])
    semantic_constrained_file = os.path.join(data_path, config["semantic_constrained_path"])
    literal_relation_file = os.path.join(data_path, config["literal_relation_path"])

    cuda_device = config["cuda_device"]
    world = config["world"]
    port = config["port"]
    sparql_endpoint_uri = config["sparql_endpoint_uri"]
    schema_retriever_uri = config["schema_retriever_uri"]
    static_dir = config["static_dir"]

    overrides = json.dumps({
        "validation_data_loader.batch_size": 1,
        "dataset_reader.lazy": True,
        "dataset_reader.semantic_constrained_file": semantic_constrained_file,
        "dataset_reader.literal_relation_file": literal_relation_file,
        # evaluate F1
        "model.evaluate_f1": True,
        # real execution
        "model.use_beam_check": True,
        # virtual execution
        "model.use_virtual_forward": False,
        # ontology-level checking
        "model.use_type_checking": False,
        # instance-level checking
        "model.use_entity_anchor": True,
        "model.decoder_beam_size": 5,
        "model.decoder_node_size": 20,
        # demo mode should be enabled
        "model.demo_mode": True,
        # at most execute 3 times
        "model.maximum_execution_times": 3
    })

    # in debug mode.
    args = [
        "--static-dir", static_dir,
        "--archive-path", model_file,
        "--predictor", "kbqa",
        "--world", world,
        "--cuda-device", cuda_device,  # -1 for CPU, device Id for GPU, default is 0
        "--port", port,
        "--schema_retriever_uri", schema_retriever_uri,
        "--sparql_endpoint_uri", sparql_endpoint_uri,
        "-o", overrides,
        "--include-package", "data_reader",
        "--include-package", "kb_parser",
        "--include-package", "kb_predictor"
    ]

    main(args)
