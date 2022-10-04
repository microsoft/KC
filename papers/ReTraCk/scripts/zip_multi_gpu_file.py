# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
import shutil
import math
import argparse


def archive_dataset_json_file(dataset_file, archive_file, batch_size, gpu_num=4):
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with open(dataset_file, mode="rb") as read_f:
        content = json.load(read_f)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",
                        type=str,
                        required=True)
    parser.add_argument("--out-dir",
                        type=str,
                        required=True)
    parser.add_argument("--batch_size_per_card",
                        type=int,
                        default=3)
    parser.add_argument("--gpu_num",
                        type=int,
                        default=4)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    archive_dataset_json_file(os.path.join(args.in_dir, "train.json"),
                              os.path.join(args.out_dir, "train"),
                              batch_size=3, gpu_num=4)
    archive_dataset_json_file(os.path.join(args.in_dir, "dev.json"),
                              os.path.join(args.out_dir, "dev"),
                              batch_size=3, gpu_num=4)
    archive_dataset_json_file(os.path.join(args.in_dir, "test.json"),
                              os.path.join(args.out_dir, "test"),
                              batch_size=3, gpu_num=4)
