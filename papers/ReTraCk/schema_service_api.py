# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from flask import Flask, request, jsonify, abort
from flask import make_response
import json
from retriever.schema_retriever.interface import DenseSchemaRetriever
import retriever.schema_retriever.dense_retriever.blink as blink

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/api/schema', methods=['POST'])
def detect():

    global model_rel, model_cls
    if not request.json or not 'question' in request.json or request.json["question"] == "" or 'world' in request.json and not request.json['world'] in ['WebQSP', 'GrailQA']:
        abort(400)
    if not 'world' in request.json:
        request.json['world'] = 'WebQSP'

    # print('Request: ')
    # print(json.dumps(request.json))

    try:
        cls_res, rel_res = loaded_retriever.dense_api(request.json['question'], request.json['world'])
        print("Complete", request.json['question'])
        return make_response(json.dumps({"Candidate Relations": rel_res, "Candidate classes": cls_res}), 200)
    except Exception as e:
        print(e)
        abort(500)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/schema_service_config.json")
    args = parser.parse_args()

    print('Using config: ' + args.config_path)

    loaded_retriever = DenseSchemaRetriever(args.config_path)

    print('Launching online service in port: ' + str(loaded_retriever.config['port']))

    app.run(debug=False,
            host='0.0.0.0',
            port=loaded_retriever.config['port'])
