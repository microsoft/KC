# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import time


def get_time_str():
    return time.strftime('_%Y_%m_%d_%H_%M_%S')


def save_log(log, model_name: str, time_str: str):
    if not time_str.startswith('_') and len(time_str):
        time_str = '_' + time_str

    file_path = '../logs/' + model_name + time_str + '_log.json'
    try:
        with open(file_path, 'w') as file:
            file.write(json.dumps(log))
    except Exception:
        print('[ERROR] save log exception, file path: ' + file_path)
