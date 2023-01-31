# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os.path

download_path = os.path.join(os.path.dirname(__file__), '../../datasets/GrailQA/domain_dict.json')
fb_domain_dict = json.load(open(file=download_path, mode="r"))
