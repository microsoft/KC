# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import sys

from allennlp.commands import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default=r"../checkpoints/grail_run_debug")
    parser.add_argument("--train_data_path", type=str, default=r"../dataset/grailqa_v9/debug.json")
    parser.add_argument("--validation_data_path", type=str, default=r"../dataset/grailqa_v9/debug.json")

    # Enable graph based matching (official metric) if the following files are provided.
    parser.add_argument("--fb_roles_file", type=str, default=r"../graph_match/fb_roles.txt")
    parser.add_argument("--fb_types_file", type=str, default=r"../graph_match/fb_types.txt")
    parser.add_argument("--reverse_properties_file", type=str, default=r"../graph_match/reverse_properties.txt")

    parser.add_argument("--literal_relation_file", type=str, default=r"../dataset/constraint/Literal_Relation.pkl")

    args = parser.parse_args()

    config_file = "configs/grail_bert.jsonnet"

    overrides = json.dumps({
        "train_data_path": args.train_data_path,
        "validation_data_path": args.validation_data_path,
        "dataset_reader.literal_relation_file": args.literal_relation_file,
        "model.fb_roles_file": args.fb_roles_file,
        "model.fb_types_file": args.fb_types_file,
        "model.reverse_properties_file": args.reverse_properties_file
    })

    # in debug mode.
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", args.serialization_dir,
        "-f",
        "--include-package", "data_reader",
        "--include-package", "kb_parser",
        "--overrides", overrides
    ]

    main()