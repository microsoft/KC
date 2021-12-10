# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from flask import Flask, request, jsonify, abort
from flask import make_response
from retriever.utils import Logger
import json
from datetime import datetime
import argparse
from retriever.kv_store_interface import KBRetriever
from retriever.configs import config_utils
import os

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/kbqa/api/v1.0/retriever', methods=['POST'])
def detect():
    if not request.json or not 'sentence' in request.json:
        abort(400)

    t1 = datetime.now()
    logger.log("Input: {}".format(json.dumps(request.json)))
    sentence = request.json["sentence"]
    output = retriever.predict(sentence)
    logger.log("Output: {}".format(json.dumps(output)))
    t2 = datetime.now()
    logger.log("Time Consumed: {}s\n".format((t2 - t1).total_seconds()))
    return jsonify(output), 201


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/retriever_config.json")
    args = parser.parse_args()

    print('Using config: ' + args.config_path)
    config = config_utils.get_config(args.config_path)

    print('Launching online service in port: ' + str(config['port']))

    retriever = KBRetriever(config)
    logger = Logger(os.path.join(config["base_data_dir"], config["log_fn"]))
    app.run(debug=False,
            host='0.0.0.0',
            port=config["port"])
