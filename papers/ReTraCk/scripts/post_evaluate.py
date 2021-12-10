# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os
import json
import shutil
import re
from allennlp.models.archival import archive_model
from tqdm import tqdm


def archive_all_checkpoint(checkpoint_folder):
    config_path = os.path.join(checkpoint_folder, "config.json")
    config_f = open(config_path, "r", encoding="utf8")
    config_content = json.load(config_f)
    # replace data reader with non-shared
    if "base_reader" in config_content["dataset_reader"]:
        config_content["dataset_reader"] = config_content["dataset_reader"]["base_reader"]
    if "distributed" in config_content:
        del config_content["distributed"]
        del config_content["validation_data_loader"]
    config_content["data_loader"] = {
        "batch_size": 1
    }
    with open(config_path, "w", encoding="utf8") as config_write_f:
        config_write_f.write(json.dumps(config_content))
    # archive all checkpoints
    temp_dir = "temp"
    abs_temp_dir = os.path.join(checkpoint_folder, temp_dir)

    if os.path.exists(abs_temp_dir):
        shutil.rmtree(abs_temp_dir)
    os.makedirs(abs_temp_dir)

    model_state_files = glob.glob(checkpoint_folder + "/best.th")

    weight_file_path = os.path.join(abs_temp_dir, "weights.th")
    shutil.copytree(os.path.join(checkpoint_folder, "vocabulary"),
                    os.path.join(abs_temp_dir, "vocabulary"))
    shutil.copy(os.path.join(checkpoint_folder, "config.json"),
                os.path.join(abs_temp_dir, "config.json"))
    for state_file in tqdm(model_state_files):
        if 'best' in state_file:
            archive_file = os.path.join(checkpoint_folder, "model.tar.gz")
        else:
            archive_file = os.path.join(checkpoint_folder, "model_{}.tar.gz".format(re.findall("\d+", state_file)[-1]))
        # remove best.th
        if os.path.exists(weight_file_path):
            os.remove(weight_file_path)
        shutil.copy(state_file, weight_file_path)
        archive_model(abs_temp_dir,
                      weights="weights.th",
                      archive_path=archive_file)


if __name__ == '__main__':
    archive_all_checkpoint("../checkpoints/ablation-entity-top9")
