# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import redis
from tqdm import tqdm
import time
import sys
import os
import pickle
import argparse

MAXBytes = 256 * 10 ** 6

def build_in_anchor_relations(in_fn, crash_cache_dir, redis_port):
    r = redis.Redis(host='localhost', port=redis_port, db=0)
    print('build connection')

    num = 0
    skip_num = 0
    done_entities = set()
    skip_entities = set()

    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(os.path.join(crash_cache_dir, "skip_entities.jsonl"),
                  mode="w", encoding="utf-8") as re_fp:

            for line in tqdm(fp):
                entity = json.loads(line.strip())
                if entity['id'] in done_entities:
                    continue
                in_relations = list(entity['one_hop_facts'].keys())
                output_line = json.dumps({
                    "id": entity['id'],
                    "in_relations": in_relations
                })

                bytes_size = sys.getsizeof(output_line)
                if bytes_size > MAXBytes:
                    skip_num += 1
                    skip_entities.add(entity['id'])
                    continue
                try:
                    r.set(entity['id'], output_line)
                    done_entities.add(entity['id'])

                except Exception as e:
                    print(e)
                    print("bytes = {} MB".format(bytes_size / 10 ** 6))
                    print(line)
                    skip_entities.add(entity['id'])
                    skip_num += 1
                    re_fp.write("{}\t{}\n".format(entity['id'], output_line))

                num += 1
                if num % 100000 == 0:
                    time.sleep(60)
                    print("processing {} lines".format(num))
            print("skip {} entities".format(skip_num))
            print("done {} entities".format(len(done_entities)))

            with open(os.path.join(crash_cache_dir, "done_entities.pkl"), mode="wb") as fp:
                pickle.dump(done_entities, fp)

            with open(os.path.join(crash_cache_dir, "skip_entities.pkl"), mode="wb") as fp:
                pickle.dump(skip_entities, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ors_file", type=str, required=True, help="path to merged_ors_per_line.json")
    parser.add_argument("--crash_cache_dir", type=str, required=True, help="path to crash_cache_dir")
    parser.add_argument("--redis_port", type=int, default=6384)
    args = parser.parse_args()
    build_in_anchor_relations(args.ors_file,
                              args.crash_cache_dir,
                              redis_port=args.redis_port)