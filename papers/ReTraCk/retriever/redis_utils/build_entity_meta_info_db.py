# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import argparse
import json
import redis
from tqdm import tqdm
import time
import os
import sys

MAXBytes = 256 * 10**6

def load_pkl(fn):
    with open(fn, mode="rb") as fp:
        pkl = pickle.load(fp)
        return pkl

def is_entity(s):
    if s.startswith('m.') or s.startswith('g.'):
        return True

def load_done_entities(fn):
    done_entities = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            entity = json.loads(line.strip())
            done_entities.add(entity['entity_id'])
        return done_entities

def dump_json_and_pkl_file(obj, jsonl_fn, pkl_fn):
    with open(pkl_fn, mode="wb") as fp:
        pickle.dump(obj, fp)
    with open(jsonl_fn, mode="w", encoding="utf-8") as fp:
        for ent in obj:
            obj[ent]["types"] = list(obj[ent]["types"])
            obj[ent]["prominent_type"] = list(obj[ent]["prominent_type"])
            obj[ent]['id'] = ent
            fp.write("{}\n".format(json.dumps(obj[ent])))

def build_freebase_ent_meta_info(entity_meta_info,
                                 out_fn,
                                 redis_port):

    with open(out_fn, mode="a", encoding="utf-8") as re_fp:
        done_entities = load_done_entities(out_fn)
        r = redis.Redis(host='localhost', port=redis_port, db=0)
        skip_num = 0
        num = 0
        for entity in tqdm(entity_meta_info):
            if entity in done_entities:
                continue
            entity_meta_info[entity]["types"] = list(entity_meta_info[entity]["types"])
            entity_meta_info[entity]["prominent_type"] = list(entity_meta_info[entity]["prominent_type"])
            sent = json.dumps(entity_meta_info[entity])
            entity_meta_info[entity]["entity_id"] = entity
            re_fp.write("{}\n".format(json.dumps(entity_meta_info[entity])))
            bytes = sys.getsizeof(sent)
            if bytes > MAXBytes:
                skip_num += 1
                continue

            try:
                r.set(entity, sent)
            except Exception as e:
                print(e)
                print("bytes = {} MB".format(bytes / 10 ** 6))
                print(sent)
                skip_num += 1
            num += 1
            if num % 100000 == 0:
                time.sleep(30)
                print("processing {} lines".format(num))
        print("skip {} entities".format(skip_num))


def run(ent_title_file,
        prominent_type_file,
        full_types_file,
        entity_desc_file,
        redis_port,
        out_dir):
    entity_title = load_pkl(ent_title_file)
    prominent_type = load_pkl(prominent_type_file)
    full_types = load_pkl(full_types_file)
    entity_desc = load_pkl(entity_desc_file)

    all_items_set = set(entity_title.keys()) | set(prominent_type.keys()) | \
                    set(full_types.keys()) | set(entity_desc.keys())
    entity_meta_info = {}
    schema_meta_info = {}
    n = 0
    for ent in all_items_set:
        n += 1
        en_label = None
        if ent in entity_title and "en" in entity_title[ent]:
            en_label = entity_title[ent]["en"]
        en_desc = None
        if ent in entity_desc and "en" in entity_desc[ent]:
            en_desc = entity_desc[ent]["en"]
        types = full_types.get(ent, [])
        p_t = prominent_type.get(ent, [])
        if is_entity(ent):
            entity_meta_info[ent] = {"en_label": en_label,
                                     "en_desc": en_desc,
                                     "prominent_type": p_t,
                                     "types": types}
        else:
            schema_meta_info[ent] = {"en_label": en_label,
                                     "en_desc": en_desc,
                                     "prominent_type": p_t,
                                     "types": types}

        if n % 1000000 == 0:
            print("processing {} lines...".format(n))

    dump_json_and_pkl_file(schema_meta_info,
                           os.path.join(out_dir, "schema_meta_info.jsonl"),
                           os.path.join(out_dir, "schema_meta_info.pkl"))

    build_freebase_ent_meta_info(entity_meta_info,
                                 os.path.join(out_dir, "entity_meta_info.jsonl"),
                                 redis_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent_title_file", type=str, required=True, help="path to freebase_title.pkl")
    parser.add_argument("--prominent_type_file", type=str, required=True, help="path to prominent_type.pkl")
    parser.add_argument("--full_types_file", type=str, required=True, help="path to freebase_types.pkl")
    parser.add_argument("--entity_desc_file", type=str, required=True, help="path to freebase_description.pkl")
    parser.add_argument("--crash_cache_dir", type=str, required=True, help="path to crash_cache_dir")
    parser.add_argument("--redis_port", type=int, default=6386)
    args = parser.parse_args()
    run(args.ent_title_file,
        args.prominent_type_file,
        args.full_types_file,
        args.entity_desc_file,
        args.redis_port,
        args.crash_cache_dir)