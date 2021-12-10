# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import pickle
import random
import shutil
from collections import Counter
from typing import List, Dict

import math
import pandas as pd
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SpacyTokenizer
from allennlp.data.tokenizers.token import Token
from tqdm import tqdm

from parser.kb_utils.kb_context import KBContext
from parser.utils import Class, Relation

ANY_STRING = "ANY"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def convert_jsonl_to_json(old_file, new_file):
    old_key = "2hop_predict_graph_query"
    new_key = "candidate_query"
    results = []
    for line in open(old_file, "r", encoding="utf8").readlines():
        obj: Dict = json.loads(line)
        obj[new_key] = obj[old_key]
        del obj[old_key]
        results.append(obj)
    with open(new_file, "w", encoding="utf8") as write_f:
        write_f.write(json.dumps(results))


def convert_prediction_jsonl_to_json(old_file, new_file):
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


def merge_semantic_constraint(constraint_file, dataset_file, write_file):
    constraint_f = open(constraint_file, "rb")
    # head: relation: tail
    head_cons_struct: Dict[str, Dict[str, List]] = pickle.load(constraint_f)
    tail_cons_struct = {}
    for head_key in head_cons_struct.keys():
        for relation_key, tail_set in head_cons_struct[head_key].items():
            for tail_key in tail_set:
                if tail_key not in tail_cons_struct:
                    tail_cons_struct[tail_key] = {}
                if relation_key not in tail_cons_struct[tail_key]:
                    tail_cons_struct[tail_key][relation_key] = set()
                tail_cons_struct[tail_key][relation_key].add(head_key)

    head_cons_frame = pd.DataFrame(head_cons_struct)
    tail_cons_frame = pd.DataFrame(tail_cons_struct)
    head_entities = set(list(head_cons_frame.columns))
    tail_entities = set(list(tail_cons_frame.columns))
    # we now directly remove relations whose tail entity is empty
    tail_relations = set(tail_cons_frame.index.values.tolist())
    with open(dataset_file, "r", encoding="utf8") as dataset_f:
        examples = json.loads(dataset_f.read())
        for i in tqdm(range(len(examples))):
            example = examples[i]
            key = "graph_query" if "candidate_query" not in example else "candidate_query"
            example_head_entities = list(set([node["class"] for node in example[key]["nodes"]]) & head_entities)
            example_tail_entities = list(set([node["class"] for node in example[key]["nodes"]]) & tail_entities)
            candidate_relations = list(set([node["relation"] for node in example[key]["edges"]]) & tail_relations)
            # fetch related subset of head entities
            head_subset_frame: pd.DataFrame = head_cons_frame.loc[candidate_relations][example_head_entities]
            relation_dict = {}
            for relation in head_subset_frame.index.values:
                relation_dict[relation] = []
                for entity_combine in head_subset_frame.loc[relation].items():
                    if pd.isna(entity_combine[1]): continue
                    start_class = entity_combine[0]
                    end_class_list = entity_combine[1]
                    relation_dict[relation].extend([(start_class, end_class) for end_class in end_class_list])
            # fetch related subset of tail entities
            # tail_subset_frame: pd.DataFrame = tail_cons_frame.loc[candidate_relations][example_tail_entities]
            # for relation in tail_subset_frame.index.values:
            #     for entity_combine in tail_subset_frame.loc[relation].items():
            #         if pd.isna(entity_combine[1]): continue
            #         end_class = entity_combine[0]
            #         start_class_list = entity_combine[1]
            #         relation_dict[relation].extend([(start_class, end_class) for start_class in start_class_list])

            for key, value in relation_dict.items():
                relation_dict[key] = list(set(value))
            examples[i]["semantic"] = relation_dict
            # back into format of {relation: [(entity_start, entity_end), (entity_start, entity_end)]...}

    with open(write_file, "w", encoding="utf8") as write_f:
        write_f.write(json.dumps(examples))


def create_ground_semantic_constraint(data_file, store_file):
    content = json.load(open(data_file, "r", encoding="utf8"))
    constraint_fake = {}
    for example in content:
        entities = example["graph_query"]["nodes"]
        relations = example["graph_query"]["edges"]
        for relation in relations:
            # fetch entity class
            relation_class = relation["relation"]
            entity_start_class = entities[relation["start"]]["class"]
            entity_end_class = entities[relation["end"]]["class"]
            if entity_start_class not in constraint_fake:
                constraint_fake[entity_start_class] = {}
            if relation_class not in constraint_fake[entity_start_class]:
                constraint_fake[entity_start_class][relation_class] = []
            if entity_end_class not in constraint_fake[entity_start_class][relation_class]:
                constraint_fake[entity_start_class][relation_class].append(entity_end_class)
    print(constraint_fake)
    with open(store_file, "wb") as store_f:
        pickle.dump(constraint_fake, store_f)


def statistic_file(file_path):
    numbers = []
    for line in open(file_path, "r", encoding="utf8").readlines():
        obj: Dict = json.loads(line)
        len_entity = len(obj["2hop_predict_graph_query"]["nodes"])
        len_relation = len(obj["2hop_predict_graph_query"]["edges"])
        numbers.append(len_entity + len_relation)
    numbers = Counter(numbers)
    print(sorted(numbers.items(), key=lambda x: x[0]))


def sample_toy_file(dataset_file, temp_file):
    with open(dataset_file, "r", encoding="utf8") as read_f:
        content = json.loads(read_f.read())
        # top100 = random.choices(content, k=120)
        # top100 = list(range(100))
        top100 = [content[1902]] * 100
    with open(temp_file, "w", encoding="utf8") as write_f:
        write_f.write(json.dumps(top100))


def count_sexpression_len_distribution(data_file):
    tokenizer = PretrainedTransformerTokenizer(model_name='bert-base-uncased')
    struct_lengths = []
    with open(data_file, "r") as data_file:
        json_obj = json.load(data_file)
        for total_cnt, ex in enumerate(json_obj):
            # TODO: now use the groundtruth as candidates, which should be prevented by shuang's candidates
            question = ex["question"]
            key = "graph_query" if "candidate_query" not in ex else "candidate_query"

            entity_list = [Class(entity_id=ins_entity["id"],
                                 entity_class=ins_entity["class"],
                                 friendly_name=ins_entity["friendly_name"],
                                 node_type=ins_entity["node_type"],
                                 node_id=ins_entity["nid"])
                           for ins_entity in ex[key]["nodes"]]
            relation_list = [Relation(relation_class=ins_relation["relation"],
                                      friendly_name=ins_relation["friendly_name"])
                             for ins_relation in ex[key]["edges"]]
            sexpression = ex["s_expression"]
            sparql_query = ex["sparql_query"]
            level = 'i.i.d.' if 'level' not in ex else ex['level']

            tokenized_utterance = tokenizer.tokenize(question)
            # tokenized_utterance = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
            #                        else Token(text=t.text, lemma_=t.text) for t in tokenized_utterance]
            # kb_context = GrailKBContext(tokenizer=tokenizer,
            #                             utterance=tokenized_utterance,
            #                             entity_list=entity_list,
            #                             relation_list=relation_list)
            # world = GrailKBWorld(kb_context=kb_context,
            #                      sparql_query=sparql_query,
            #                      sexpression=sexpression,
            #                      origin_utterance=question,
            #                      answers=answers,
            #                      level=level)
            # sexpression_struct = world.language.logical_form_to_action_sequence(world.sexpression_parse)
            struct_lengths.append(len(tokenized_utterance))

    counter = Counter(struct_lengths)
    print(sorted(counter.items(), key=lambda x: x[0]))


def archive_dataset_file(dataset_file, archive_file, batch_size, gpu_num=4):
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with open(dataset_file, "r", encoding="utf8") as read_f:
        content = json.loads(read_f.read())
        content_len = len(content)
        borders = [0]
        rough_size = math.ceil(content_len / (batch_size * gpu_num))
        fill_size = rough_size * batch_size * gpu_num - content_len
        # fill the content
        fill_content = random.choices(content, k=fill_size)
        content = content + fill_content

        content_len = len(content)
        chunk_size = math.ceil(content_len / gpu_num)
        for i in range(gpu_num):
            if i == gpu_num - 1:
                borders.append(content_len)
            else:
                borders.append(borders[-1] + chunk_size)
        for i in range(gpu_num):
            chunk_content = content[borders[i]: borders[i + 1]]
            last_path = os.path.split(dataset_file)[-1]
            chunk_file_name = last_path.strip(".json") + str(i) + ".json"
            file_path = os.path.join(temp_dir, chunk_file_name)
            with open(file_path, "w", encoding="utf8") as write_f:
                write_f.write(json.dumps(chunk_content))

    # archive files
    shutil.make_archive(archive_file, 'zip', temp_dir)
    shutil.rmtree(temp_dir)


def archive_dataset_jsonl_file(dataset_file, archive_file, batch_size, gpu_num=4):
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with open(dataset_file, "r", encoding="utf8") as read_f:
        # content = json.loads(read_f.read())
        content = []
        for line in read_f:
            content.append(json.loads(line.strip()))
        content_len = len(content)
        borders = [0]
        rough_size = math.ceil(content_len / (batch_size * gpu_num))
        fill_size = rough_size * batch_size * gpu_num - content_len
        # fill the content
        fill_content = random.choices(content, k=fill_size)
        content = content + fill_content

        content_len = len(content)
        chunk_size = math.ceil(content_len / gpu_num)
        for i in range(gpu_num):
            if i == gpu_num - 1:
                borders.append(content_len)
            else:
                borders.append(borders[-1] + chunk_size)
        for i in range(gpu_num):
            chunk_content = content[borders[i]: borders[i + 1]]
            last_path = os.path.split(dataset_file)[-1]
            chunk_file_name = last_path.strip(".json") + str(i) + ".json"
            file_path = os.path.join(temp_dir, chunk_file_name)
            with open(file_path, "w", encoding="utf8") as write_f:
                write_f.write(json.dumps(chunk_content))

    # archive files
    shutil.make_archive(archive_file, 'zip', temp_dir)
    shutil.rmtree(temp_dir)


def count_entity_len_distribution(data_file):
    tokenizer = SpacyTokenizer()
    # model_name = 'bert-base-uncased')
    struct_lengths = []
    with open(data_file, "r") as data_file:
        json_obj = json.load(data_file)
        for total_cnt, ex in enumerate(json_obj):
            # TODO: now use the groundtruth as candidates, which should be prevented by shuang's candidates
            question = ex["question"]
            # key = "graph_query" if "candidate_query" not in ex else "candidate_query"
            key = "graph_query"
            entity_list = [Class(entity_id=ins_entity["id"],
                                 entity_class=ins_entity["class"],
                                 friendly_name=ins_entity["friendly_name"],
                                 node_type=ins_entity["node_type"],
                                 node_id=ins_entity["nid"])
                           for ins_entity in ex[key]["nodes"]]
            relation_list = [Relation(relation_class=ins_relation["relation"],
                                      friendly_name=ins_relation["friendly_name"])
                             for ins_relation in ex[key]["edges"]]
            tokenized_utterance = tokenizer.tokenize(question)
            tokenized_utterance = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
                                   else Token(text=t.text, lemma_=t.text) for t in tokenized_utterance]
            kb_context = KBContext(tokenizer=tokenizer,
                                        utterance=tokenized_utterance,
                                        entity_list=entity_list,
                                        encode_method="all_domain",
                                        relation_list=relation_list)
            ind = 0
            for entity_token in kb_context.entity_tokens:
                struct_lengths.append(len(entity_token))
                ind += 1
    counter = Counter(struct_lengths)
    print(sorted(counter.items(), key=lambda x: x[0]))


def filter_relation(semantic_constraint_file):
    constraint_f = open(semantic_constraint_file, "rb")
    # head: relation: tail
    head_cons_struct: Dict[str, Dict[str, List]] = pickle.load(constraint_f)
    relation_dict = {}
    for head_key in head_cons_struct.keys():
        if head_key == '':
            continue
        for relation_key, tail_set in head_cons_struct[head_key].items():
            if relation_key not in relation_dict:
                relation_dict[relation_key] = set()
            for tail_key in tail_set:
                if tail_key != '' and '.' in tail_key:
                    relation_dict[relation_key].add((head_key, tail_key))
    print("Collect all valid relation keys: {}".format(len(relation_dict.keys())))
    keep_relation_dict = {}
    for rel_key in tqdm(relation_dict.keys()):
        new_rel_tup = []
        rel_tup = relation_dict[rel_key]
        if len(rel_tup) == 0: continue
        # try to reduce the space of relations
        head_entities = Counter([tup[0] for tup in rel_tup])
        tail_entities = Counter([tup[1] for tup in rel_tup])
        reduce_head_set = [tup[0] for tup in head_entities.most_common() if tup[1] >= 100]
        reduce_tail_set = [tup[0] for tup in tail_entities.most_common() if tup[1] >= 100]
        if len(reduce_head_set) > 1000 and len(reduce_tail_set) > 1000:
            # do not add this to semantic constraint (no meaning)
            continue
        count_total_num = 0
        for tup in rel_tup:
            if tup[0] not in reduce_head_set and tup[1] not in reduce_tail_set:
                new_rel_tup.append(tup)
                count_total_num += 1
            # if count_total_num > 2000: break
        # pass
        # if count_total_num > 2000:
        #     continue
        for head in reduce_head_set:
            new_rel_tup.append((head, ANY_STRING))
        for tail in reduce_tail_set:
            new_rel_tup.append((ANY_STRING, tail))
        keep_relation_dict[rel_key] = set(new_rel_tup)

    keep_relation_dict = keep_relation_dict
    print("Keep valid relation keys: {}".format(len(keep_relation_dict.keys())))

    write_file = semantic_constraint_file.replace(".pkl", ".filter.pkl")
    with open(write_file, "wb") as write_f:
        pickle.dump(keep_relation_dict, write_f)
    # numbers = Counter(lens)
    # sort_numbers = sorted(numbers.items(), key=lambda x: x[0])
    # for tup in sort_numbers:
    #     print("{}\t{}".format(tup[0], tup[1]))


def extract_terminals_from_webqsp(file_path):
    defined_predicates = ['ARGMAX', 'ARGMIN', 'lt', 'le', 'gt', 'ge', 'AND', 'JOIN', 'R', ')', 'COUNT']
    new_examples = []
    with open(file_path, "r", encoding="utf8") as read_f:
        examples = json.load(read_f)
        for example in examples:
            sexpression_flatten = example["s_expression"]
            if sexpression_flatten is None:
                example['kb_items'] = []
                new_examples.append(example)
                continue
            sexpression_strut = sexpression_flatten.split()
            ind = len(sexpression_strut) - 1
            while ind >= 0:
                if sexpression_strut[ind] in defined_predicates or sexpression_strut[ind][0] == '(':
                    del sexpression_strut[ind]
                else:
                    sexpression_strut[ind] = sexpression_strut[ind].rstrip(')')
                    ind -= 1
            example['kb_items'] = sexpression_strut
            new_examples.append(example)
    write_f = open(file_path.replace(".json", ".kb.json"), "w", encoding="utf8")
    write_f.write(json.dumps(new_examples))
    write_f.close()


def count_valid_answer(jsonl_file):
    answer_len = []
    with open(jsonl_file, "r", encoding="utf8") as jsonl_f:
        result = json.load(jsonl_f)
        for example in result.values():
            answer = example["answer"]
            answer_len.append(len(answer))
    counter = Counter(answer_len)
    print(sorted(counter.items(), key=lambda x: x[0]))


def analyze_difference(debug_file_1, debug_file_2):
    debug_content_1 = open(debug_file_1, "r", encoding="utf8").readlines()
    debug_content_2 = open(debug_file_2, "r", encoding="utf8").readlines()
    new_file = open("different.jsonl", "w", encoding="utf8")
    for debug_line_1, debug_line_2 in zip(debug_content_1, debug_content_2):
        debug_obj_1 = json.loads(debug_line_1)
        debug_obj_2 = json.loads(debug_line_2)
        if debug_obj_1["correct"] and not debug_obj_2["correct"]:
            debug_obj_1["type_predict_sexpression"] = debug_obj_2["predict_sexpression"]
            new_file.write(json.dumps(debug_obj_1) + "\n")
    new_file.close()


def analyze_analysis_difference(debug_file_1, debug_file_2):
    debug_content_1 = open(debug_file_1, "r", encoding="utf8").readlines()
    debug_content_2 = open(debug_file_2, "r", encoding="utf8").readlines()
    dataset_content = open("../dataset/grailqa_v9/dev_v9_pipeline_top3.json").read()
    dict_map = {}
    for example in json.loads(dataset_content):
        dict_map[example["qid"]] = example["level"]
    new_file = open("different.jsonl", "w", encoding="utf8")
    for debug_line_1, debug_line_2 in zip(debug_content_1, debug_content_2):
        debug_obj_1 = json.loads(debug_line_1)
        debug_obj_2 = json.loads(debug_line_2)
        if debug_obj_1["logic_form_correct"] != debug_obj_2["logic_form_correct"]:
            debug_obj_1["lookahead_predict_sexpression"] = debug_obj_2["predict_sexpression"]
            debug_obj_1["lookahead_logic_form_correct"] = debug_obj_2["logic_form_correct"]
            debug_obj_1["level"] = dict_map[debug_obj_1["qid"]]
            new_file.write(json.dumps(debug_obj_1) + "\n")
    new_file.close()


def merge_origin_and_process(origin_file, processed_file):
    def try_fix_year_tag(node_info):
        if node_info["id"].endswith("#gYear"):
            replace_id = node_info["id"].replace("^^http://www.w3.org/2001/XMLSchema#gYear", "")
            node_info["id"] = replace_id
        return node_info

    def add_special_class(class_name):
        return {
            "id": class_name,
            "class": class_name,
            "node_type": "class",
            "friendly_name": class_name,
            "nid": 0
        }

    origin_dict = {}
    with open(origin_file, "r", encoding="utf8") as read_f:
        origin_content = json.loads(read_f.read())
        for example in origin_content:
            origin_dict[example["qid"]] = example
    with open(processed_file, "r", encoding="utf8") as process_f:
        process_content = json.loads(process_f.read())
    new_file = processed_file.replace(".json", ".fix.json")
    write_f = open(new_file, "w", encoding="utf8")
    results = []
    is_training = "train" in processed_file
    # is_training = True
    drop_count = 0
    null_sexpression = 0
    special_case = ["male", "female", "Country", "State"]
    for process_obj in process_content:
        origin_obj = origin_dict[process_obj["sent_idx_unq"]]
        assert origin_obj["qid"] == process_obj["sent_idx_unq"]
        process_obj["sparql_query"] = origin_obj["sparql_query"]
        process_obj["answer"] = origin_obj["answer"]
        process_obj["s_expression"] = origin_obj["s_expression"]

        if origin_obj["s_expression"] is not None:
            null_sexpression += 1
            continue

        process_obj["graph_query"] = {
            "nodes": [],
            "edges": []
        }
        for class_name in special_case:
            process_obj["graph_query"]["nodes"].append(
                add_special_class(class_name)
            )
            process_obj["candidate_query"]["nodes"].append(
                add_special_class(class_name)
            )

        if is_training:
            # construct graph_query
            candidate_nodes = {}
            for node in process_obj["candidate_query"]["nodes"]:
                node = try_fix_year_tag(node)
                candidate_nodes[node["id"]] = node
            candidate_edges = {}
            for node in process_obj["candidate_query"]["edges"]:
                candidate_edges[node["relation"]] = node
            kb_items = [kb_item for kb_item in origin_obj["kb_items"] if
                        kb_item not in special_case and kb_item != "NOW"]
            for kb_item in kb_items:
                if kb_item in candidate_nodes:
                    process_obj["graph_query"]["nodes"].append(candidate_nodes[kb_item])
                elif kb_item in candidate_edges:
                    process_obj["graph_query"]["edges"].append(candidate_edges[kb_item])
                else:
                    break
            if len(process_obj["graph_query"]["nodes"]) + len(process_obj["graph_query"]["edges"]) == len(
                    kb_items) + len(special_case):
                results.append(process_obj)
            else:
                drop_count += 1
        else:
            results.append(process_obj)
    json.dump(results, write_f)
    print("Drop {} cases because no ground-truth".format(drop_count))
    print("Empty sexpression is: {}".format(null_sexpression))
    write_f.close()


if __name__ == '__main__':
    # merge_origin_and_process("../dataset/resource/webqsp_test_www.kb.json",
    #                          "../dataset/webqsp_v10/webqsp_test_v10_top5.json")
    # analyze_analysis_difference("../checkpoints/grailqa_best/analysis/dev_v9_pipeline_top3instance_no_literal.jsonl",
    #                             "../checkpoints/grailqa_best/analysis/dev_v9_pipeline_top3instance.jsonl")
    # analyze_difference("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_grammar.jsonl",
    #                    "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_instance.jsonl")
    # convert_prediction_jsonl_to_json("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\final_fix.jsonl",
    #                                  "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\final_fix.json")
    # convert_prediction_jsonl_to_json("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_instance.jsonl",
    #                                  "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_instance.json")
    # convert_prediction_jsonl_to_json("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_real.jsonl",
    #                                  "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_real.json")
    # convert_prediction_jsonl_to_json("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_virtual.jsonl",
    #                                  "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\only_virtual.json")
    # convert_prediction_jsonl_to_json("..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\virtual+real.jsonl",
    #                                  "..\\test_files\\dev-v9-top3-b60-lr-2e-5-shuffle\\virtual+real.json")
    # convert_prediction_jsonl_to_json("..\\test_files\\submit_v4_nothing_authorel.jsonl",
    #                                  "..\\test_files\\submit_v4_nothing_authorel.json")
    # extract_terminals_from_webqsp("..\\OtherData\\webqsp_test_www.json")
    # convert_jsonl_to_json("..\\share_data\\grailqa_dense_commons_v2_topk_anchor_relations\\train_dense_v0_oracle_entities_top1_ent.jsonl",
    #                       "..\\dataset\\train_v9_pipeline_oracle_top1.json")
    # convert_jsonl_to_json("..\\share_data\\grailqa_dense_commons_v2_topk_anchor_relations\\dev_dense_v0_bootleg_prior_entities_top1_ent.jsonl",
    #                       "..\\dataset\\dev_v9_pipeline_top1.json")
    # convert_jsonl_to_json("..\\share_data\\grailqa_dense_commons_v2_topk_anchor_relations\\test_dense_v0_bootleg_prior_entities_top1_ent.jsonl",
    #                       "..\\dataset\\test_v9_pipeline_top1.json")

    # statistic_file("..\\share_data\\train_v1_aligned_bert_ner_datetime_prior_baseline_2hop_filter_v2.json")
    # create_ground_semantic_constraint("..\\dataset\\train_v1.json", "..\\dataset\\ground.pkl")
    # merge_semantic_constraint("..\\dataset\\ground.pkl",
    #                           "..\\dataset\\train_v1.json",
    #                           "..\\dataset\\train_v1.sem.json")
    # merge_semantic_constraint("..\\share_data\\spoType_final.pkl",
    #                           "..\\dataset\\dev_v1_pipeline.json",
    #                           "..\\dataset\\dev_v1_pipeline.simple.sem.json")
    # sample_toy_file("..\\dataset\\grailqa_v9\\dev_v9_pipeline_top3.json",
    #                 "..\\dataset\\grailqa_v9\\debug.json")
    # shutil.copy("..\\dataset\\train_v4_pipeline_158.json",
    #             "..\\dataset\\dev_v4_pipeline_158.json")
    # archive_dataset_file("..\\dataset\\train_v9_pipeline_oracle_top1.json",
    #                      "..\\dataset\\train_v9_pipeline_oracle_top1",
    #                      batch_size=3)
    # archive_dataset_file("..\\dataset\\dev_v9_pipeline_top1.json",
    #                      "..\\dataset\\dev_v9_pipeline_top1",
    #                      batch_size=3)
    # merge_semantic_constraint("..\\share_data\\spoType_final.pkl",
    #                           "..\\dataset\\case_133.json",
    #                           "..\\dataset\\case_133.sem.json")
    # count_entity_len_distribution("..\\dataset\\dev_v8_pipeline_top3.json")
    # count_entity_len_distribution("..\\dataset\\train_v8_pipeline_oracle_top3.json")

    # count_valid_answer("..\\test_files\\retrack_v3.json")
    # count_entity_len_distribution("..\\dataset\\train_v8_pipeline_oracle_top3.json")
    # count_entity_len_distribution("..\\dataset\\train_v8_pipeline_oracle_top3.json")
    # count_entity_len_distribution("..\\dataset\\test_v8_pipeline_top3.json")

    # archive_dataset_file("..\\dataset\\train_v4_pipeline_oracle.json",
    #                      "..\\dataset\\train_v4_pipeline_158")
    # archive_dataset_file("..\\dataset\\train_v4_pipeline_160.json",
    #                      "..\\dataset\\train_v4_pipeline_160")
    # filter_relation("../share_data/types_new_spo.pkl")
    # analyze_difference("debug_grammar.jsonl", "debug_type.jsonl")

    # in_dir = r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\grailqa_merged_ngram_ner_genre"
    # zip_dir = r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\grailqa_merged_ngram_ner_genre\zip"
    # for k in [1, 3, 5]:
    #     archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_train_input_top{}.jsonl".format(k)),
    #                                os.path.join(zip_dir, "grailqa_train_input_top{}_v12.zip".format(k)),
    #                                batch_size=3, gpu_num=4)
    #     archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_dev_input_top{}.jsonl".format(k)),
    #                                os.path.join(zip_dir, "grailqa_dev_input_top{}_v12.zip".format(k)),
    #                                batch_size=3, gpu_num=4)
    #     archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_test_input_top{}.jsonl".format(k)),
    #                                os.path.join(zip_dir, "grailqa_test_input_top{}_v12.zip".format(k)),
    #                                batch_size=3, gpu_num=4)

    in_dir = r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\merged_min_len4"
    zip_dir = r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\merged_min_len4\zip"
    for k in [1, 3, 5]:
        archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_train_input_top{}.jsonl".format(k)),
                                   os.path.join(zip_dir, "grailqa_train_input_top{}_v13".format(k)),
                                   batch_size=3, gpu_num=4)
        archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_dev_input_top{}.jsonl".format(k)),
                                   os.path.join(zip_dir, "grailqa_dev_input_top{}_v13".format(k)),
                                   batch_size=3, gpu_num=4)
        archive_dataset_jsonl_file(os.path.join(in_dir, "grailqa_test_input_top{}.jsonl".format(k)),
                                   os.path.join(zip_dir, "grailqa_test_input_top{}_v13".format(k)),
                                   batch_size=3, gpu_num=4)

    # archive_dataset_jsonl_file(r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\zhiwei_cross_validation\grailqa_train_input_top3.jsonl",
    #                            r"E:\users\v-shuanc\Workspace\GrailQA\Data\Summary\Parser\zhiwei_cross_validation\zip\grailqa_train_input_top3.zip",
    #                            batch_size=3, gpu_num=4)