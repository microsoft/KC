import argparse
import json
import sys
import os
from allennlp.commands import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default=r"")
    parser.add_argument("--train_data_path", type=str, default=r"")
    parser.add_argument("--validation_data_path", type=str, default=r"")
    parser.add_argument("--fb_roles_file", type=str, default=r"")
    parser.add_argument("--fb_types_file", type=str, default=r"")
    parser.add_argument("--reverse_properties_file", type=str, default=r"")
    parser.add_argument("--literal_relation_file", type=str, default=r"../dataset/constraint/Literal_Relation.pkl")

    args = parser.parse_args()

    config_file = "configs/grail_bert_distributed_aml.jsonnet"

    overrides = json.dumps({
        "train_data_path": args.train_data_path,
        "validation_data_path": args.validation_data_path,
        "dataset_reader.base_reader.literal_relation_file": args.literal_relation_file,
        "model.fb_roles_file": args.fb_roles_file,
        "model.fb_types_file": args.fb_types_file,
        "model.reverse_properties_file": args.reverse_properties_file
    })

    # if os.path.exists(args.serialization_dir):
    #     recover_option = True
    # else:
    #     recover_option = False
    recover_option = False

    if recover_option:
        sys.argv = [
            "allennlp",  # command name, not used by main
            "train",
            config_file,
            "-s", args.serialization_dir,
            #"-f",
            "--include-package", "data_reader",
            "--include-package", "kb_parser",
            "-r",
            "--overrides", overrides,
            "--file-friendly-logging"
        ]
    else:
        sys.argv = [
            "allennlp",  # command name, not used by main
            "train",
            config_file,
            "-s", args.serialization_dir,
            "-f",
            "--include-package", "data_reader",
            "--include-package", "kb_parser",
            "--overrides", overrides,
            "--file-friendly-logging"
        ]

    main()