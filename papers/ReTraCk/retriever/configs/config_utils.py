# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os


def get_config(config_path):
    with open(config_path, mode='r') as fp:
        config = json.load(fp)
        config = process_config_paths(config)
        return config


def process_config_paths(config):

    # Read base environment variables
    data_path = os.environ.get("DATA_PATH")
    retrack_path = os.environ.get("RETRACK_HOME")

    if retrack_path is None:
        print("Warning: RETRACK_HOME environment variable not set.")

    if data_path is None:
        print("Warning: DATA_PATH environment variable not set.")
        return config

    if isinstance(config, dict):
        pairs = config.items()
    else:
        print("Warming: Config must be a dict. " + type(config))
        return config

    for key, value in pairs:
        if isinstance(value, str):
            if value.startswith('./') or value.startswith('.\\'):
                path = data_path
                if '/configs/' in value or '\\configs\\' in value or '/static' in value or '\\static' in value:
                    path = retrack_path
                config[key] = value.replace('./', path + '/').replace('.\\', path + '\\')
        elif isinstance(value, dict):
            config[key] = process_config_paths(value)

    config["base_data_dir"] = data_path

    return config


if __name__ == '__main__':

    config_path = "./configs/schema_service_config.json"
    config = get_config(config_path)
