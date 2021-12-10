# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from allennlp.commands import main
import json
import argparse
from typing import Dict
from repack_aml_model import modify_config_and_pack, unzip_cat_file
from retriever.configs import config_utils


def convert_test_jsonl_to_json(old_file, new_file):

    results = {}
    for line in open(old_file, "r", encoding="utf8").readlines():
        obj: Dict = json.loads(line)
        for logical_form, answer, qid in zip(obj["logical_form"], obj["answer"], obj["qid"]):
            results[qid] = {
                "logical_form": logical_form,
                "answer": answer
            }

    with open(new_file, "w", encoding="utf8") as write_f:
        write_f.write(json.dumps(results))


def gen_log_file_name(evaluate_file_name,
                      use_beam_check=False,
                      use_virtual_forward=False,
                      use_type_checking=False,
                      use_entity_anchor=False):

    file_name = os.path.split(evaluate_file_name)[-1].split('.')[0] + "_"

    if use_beam_check:
        file_name += "beam_check_"
    if use_virtual_forward:
        file_name += "virtual_exec_"
    if use_type_checking:
        file_name += "ontology_"
    if use_entity_anchor:
        file_name += "instance_"

    file_name = file_name.strip("_")
    return file_name


if __name__ == '__main__':

    # Read base environment variables
    data_path = os.environ.get("DATA_PATH")

    if data_path is None:
        print("Warning: DATA_PATH environment variable not set.")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, default="../retrack/configs/parser_eval_grailqa_dev.json")
    cli_args = arg_parser.parse_args()

    print('Using config: ' + cli_args.config_path)

    # Load config
    config = config_utils.get_config(cli_args.config_path)

    model_file = os.path.join(data_path, config["model_file"])
    eval_file = os.path.join(data_path, config["eval_file"])
    roles_file = os.path.join(data_path, config["roles_file"])
    types_file = os.path.join(data_path, config["types_file"])
    reverse_properties_file = os.path.join(data_path, config["reverse_properties_file"])
    semantic_constraints_file = os.path.join(data_path, config["semantic_constraints_file"])
    literal_relations_file = os.path.join(data_path, config["literal_relations_file"])

    out_dir = config["output_path"]
    if out_dir.startswith('./') or out_dir.startswith('.\\'):
        out_dir = os.path.join(data_path, out_dir)

    dataset_name = config["dataset_name"]
    use_beam_check = config["use_beam_check"]
    use_virtual_forward = config["use_virtual_forward"]
    use_type_checking = config["use_type_checking"]
    use_entity_anchor = config["use_entity_anchor"]
    eval_f1 = config["eval_f1"]
    eval_hits1 = config["eval_hits1"]

    args = [
        "--evaluate_file", eval_file,
        "--model_file", model_file,
        "--fb_roles_file", roles_file,
        "--fb_types_file", types_file,
        "--reverse_properties_file", reverse_properties_file,
        "--semantic_constrained_file", semantic_constraints_file,
        "--literal_relation_file", literal_relations_file,
        "--use_beam_check", use_beam_check,
        "--use_virtual_forward", use_virtual_forward,
        "--use_type_checking", use_type_checking,
        "--use_entity_anchor", use_entity_anchor,
        "--evaluate_f1", eval_f1,
        "--evaluate_hits1", eval_hits1,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_file", type=str, default=r"./Dataset/GrailQA/processed/dev.json")
    parser.add_argument("--model_file", type=str, default=r"./Parser/grailqa_model.tar.gz")

    parser.add_argument("--fb_roles_file", type=str, default=r"./Parser/graph_match/fb_roles.txt")
    parser.add_argument("--fb_types_file", type=str, default=r"./Parser/graph_match/fb_types.txt")
    parser.add_argument("--reverse_properties_file", type=str, default=r"./Parser/graph_match/reverse_properties.txt")

    parser.add_argument("--semantic_constrained_file", type=str, default=r"./Parser/constraint/SRO_COM_SC.filter.pkl")
    parser.add_argument("--literal_relation_file", type=str, default=r"./Parser/constraint/Literal_Relation.pkl")

    parser.add_argument("--use_beam_check", type=bool, default=False)
    parser.add_argument("--use_virtual_forward", type=bool, default=False)
    parser.add_argument("--use_type_checking", type=bool, default=False)
    parser.add_argument("--use_entity_anchor", type=bool, default=False)
    parser.add_argument("--evaluate_f1", type=bool, default=False)
    parser.add_argument("--evaluate_hits1", type=bool, default=False)

    args = parser.parse_args(args)

    evaluate_file = args.evaluate_file
    log_file_name = gen_log_file_name(args.evaluate_file,
                                      args.use_beam_check,
                                      args.use_virtual_forward,
                                      args.use_type_checking,
                                      args.use_entity_anchor)

    analysis_dir = os.path.join(out_dir, f'{dataset_name}_{log_file_name}_analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    overrides = json.dumps({
        "model.fb_roles_file": args.fb_roles_file,
        "model.fb_types_file": args.fb_types_file,
        "model.reverse_properties_file": args.reverse_properties_file,
        "model.log_eval_info_file": os.path.join(analysis_dir, "{}.md".format(log_file_name)),
        "model.log_analysis_info_file": os.path.join(analysis_dir, "{}.jsonl".format(log_file_name)),
        # if you want to use lazy mode, please check the config file and make sure the validation_data_loader looks like
        "validation_data_loader.batch_size": 1,
        "dataset_reader.lazy": True,
        "dataset_reader.semantic_constrained_file": args.semantic_constrained_file,
        "dataset_reader.literal_relation_file": args.literal_relation_file,
        # evaluate F1
        "model.evaluate_f1": args.evaluate_f1,
        # evaluate hits1
        "model.evaluate_hits1": args.evaluate_hits1,
        # real execution
        "model.use_beam_check": args.use_beam_check,
        # virtual execution
        "model.use_virtual_forward": args.use_virtual_forward,
        # ontology-level checking
        "model.use_type_checking": args.use_type_checking,
        # instance-level checking
        "model.use_entity_anchor": args.use_entity_anchor,
        "model.decoder_beam_size": 5,
        "model.decoder_node_size": 20,
        "model.maximum_execution_times": 3
    })

    evaluate_file_name = os.path.split(args.evaluate_file)[-1]
    # produce metric file
    output_options = ["--predictions-output-file", os.path.join(analysis_dir, "{}_output.jsonl".format(log_file_name)),
                      "--output-file", os.path.join(analysis_dir, "{}.json".format(log_file_name))]

    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        *output_options,
        "--cuda-device", "0",
        "-o", overrides,
        "--include-package", "data_reader",
        "--include-package", "kb_parser",
        args.model_file,
        evaluate_file
    ]

    print('Launching eval...')

    main()

    print('Converting format...')
    jsonl_file = os.path.join(analysis_dir, "{}_output.jsonl".format(log_file_name))
    submit_json_file = os.path.join(analysis_dir, "{}_output.json".format(log_file_name))

    # convert the original json line file into a json file to submit
    print(f'Generating output at: {analysis_dir}')
    convert_test_jsonl_to_json(jsonl_file, submit_json_file)
    os.remove(jsonl_file)
