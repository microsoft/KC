# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A `Flask <https://palletsprojects.com/p/flask/>`_ server for serving predictions
from a single AllenNLP model. It also includes a very, very bare-bones
web front-end for exploring predictions (or you can provide your own).

For example, if you have your own predictor and model in the `my_stuff` package,
and you want to use the default HTML, you could run this like

```
python -m allennlp.service.server_simple \
    --archive-path allennlp/tests/fixtures/bidaf/serialization/model.tar.gz \
    --predictor machine-comprehension \
    --title "Demo of the Machine Comprehension Text Fixture" \
    --field-name question --field-name passage
```
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Callable, Optional

from allennlp.common import JsonDict
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from flask import Flask, request, Response, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from sparql_executor import retrieve_ent_meta_info, retrieve_schema_meta_info, set_sparql_wrapper
import html
from utils import extract_sxpression_structure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict


def render_entity_template(entity_list):
    entity_template = """
    <a tabindex="{index}"
       class="btn popover-dismiss {style}"
       role="button"
       data-mdb-placement="top"
       data-mdb-toggle="popover"
       data-mdb-trigger="hover"
       data-mdb-html=true
       data-mdb-original-title="{title}"
       data-mdb-content="{desc}">
        {text}
    </a>
    """
    desc_template = "<strong>Identifier</strong>: {0}<br/> <strong>Notable Type</strong>: {1}<br/>"
    link_template = "<span>(<a href=\"{0}\" target=\"_blank\" class=\"stretched-link\">Detail</a>) &nbsp;</span>"
    template = ""
    for entity in entity_list:
        if entity["type"] != "text":
            style = "btn-outline-dark"
            if entity["select"] is True and entity["type"] == "entity":
                style = "btn-outline-secondary"
            elif entity["select"] is True and entity["type"] == "literal":
                style = "btn-outline-primary"

            template += entity_template.format(
                index=entity["index"],
                title=html.escape(entity["title"]),
                desc=html.escape(desc_template.format(entity["id"], entity["ent_type"])),
                text=html.escape(entity["text"]),
                style=style
            ) + "\n"

            if entity["type"] == "entity":
                template += link_template.format(entity["link"])
        else:
            template += entity["text"] + " "

    return template


def render_schema_template(schema_list):
    schema_template = """
    <div class="table-row {style}">
        <div class="serial">{index}</div>
        <div class="schema">
            <span>{name} <a href="{link}" target="_blank" class="stretched-link">Detail</a></span>
        </div>
        <div class="percentage">
            <div class="progress">
                <div class="progress-bar color-1" role="progressbar"
                     aria-valuenow="{score}" aria-valuemin="0"
                     style="width: {score}%"
                     aria-valuemax="100"></div>
            </div>
        </div>
    </div>
    """
    template = ""
    for index, schema_item in enumerate(schema_list):
        template += schema_template.format(
            index=index + 1,
            name=schema_item["name"],
            score=schema_item["score"],
            style="selected" if schema_item["select"] else "",
            link=schema_item["link"]
        ) + "\n"

    return template


def render_logic_form_template(logic_form):
    val_template = '<a href="#" class="btn btn-default">{}</a>'

    def recursive_format(node: List):
        cur_val = "<li>" + val_template.format(node[0]) + "<ul>"
        for i in range(1, len(node)):
            if isinstance(node[i], str):
                node_val = "<li>" + val_template.format(node[i]) + "</li>"
            else:
                node_val = recursive_format(node[i])
            cur_val += node_val

        cur_val += "</ul></li>"

        return cur_val

    if logic_form != '':
        logic_form_strut = extract_sxpression_structure(logic_form)
        logic_form_html = "<ul>" + recursive_format(logic_form_strut) + "</ul>"
    else:
        logic_form_html = ""

    return logic_form_html


def render_answer_template(answer_list):
    template = ""
    entity_template = """
    <a tabindex="{index}"
       class="btn popover-dismiss {style}"
       role="button"
       data-mdb-placement="top"
       data-mdb-toggle="popover"
       data-mdb-trigger="hover"
       data-mdb-html=true
       data-mdb-original-title="{title}"
       data-mdb-content="<strong>Identifier</strong>: {id}<br/> <strong>Notable Type</strong>: {type}<br/>">
        {title}
    </a>
    """

    if len(answer_list):
        meta_info, _ = retrieve_ent_meta_info(answer_list)
        for ind, answer in enumerate(answer_list):
            if answer in meta_info and meta_info[answer]["ent_name"]:
                style = "btn-outline-secondary"
                entity = meta_info[answer]
                link = "<span>(<a href=\"/meta/{0}\" target=\"_blank\" class=\"stretched-link\">Detail</a>) &nbsp;</span>".format(answer)
                template += entity_template.format(
                    index=ind,
                    title=html.escape(entity["ent_name"]),
                    id=html.escape(answer),
                    type=html.escape(entity["ent_type"]),
                    style=style
                ) + "\n" + link + "\n"
            else:
                style = "btn-outline-primary"
                template += entity_template.format(
                    index=ind,
                    title=html.escape(answer),
                    id=html.escape(answer),
                    type="literal",
                    style=style
                ) + "\n"
    else:
        template = "<p>{}</p>".format("Sorry, No Answers Found")
    return template


def make_app(predictor: Predictor,
             field_names: List[str] = None,
             static_dir: str = None,
             sanitizer: Callable[[JsonDict], JsonDict] = None) -> Flask:
    """
    Creates a Flask app that serves up the provided ``Predictor``
    along with a front-end for interacting with it.

    If you want to use the built-in bare-bones HTML, you must provide the
    field names for the inputs (which will be used both as labels
    and as the keys in the JSON that gets sent to the predictor).

    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``. In that case you
    don't need to supply the field names -- that information should
    be implicit in your demo site. (Probably the easiest thing to do
    is just start with the bare-bones HTML and modify it.)

    In addition, if you want somehow transform the JSON prediction
    (e.g. by removing probabilities or logits)
    you can do that by passing in a ``sanitizer`` function.
    """
    if static_dir is not None:
        static_dir = os.path.abspath(static_dir)
        if not os.path.exists(static_dir):
            logger.error("app directory %s does not exist, aborting", static_dir)
            sys.exit(-1)
    elif static_dir is None and field_names is None:
        print("Neither build_dir nor field_names passed. Demo won't render on this port.\n"
              "You must use nodejs + react app to interact with the server.")

    app = Flask(__name__, template_folder=static_dir)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file(os.path.join(static_dir, 'index.html'))

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_json(data)["demo"]

        render_entity = render_entity_template(prediction["entity_candidate"])
        render_relation = render_schema_template(prediction["schema_relation"])
        render_class = render_schema_template(prediction["schema_class"])
        render_answer = render_answer_template(prediction["answer"])
        render_logic = render_logic_form_template(prediction["logic_form"])
        render_sparql = html.escape(prediction["sparql"]).replace("\n", "<br/>")

        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify({
            "answer": render_answer,
            "entity": render_entity,
            "relation": render_relation,
            "class": render_class,
            "logic_form": render_logic,
            "sparql": render_sparql,
            "original_logic_form": prediction["logic_form"]
        })

    @app.route('/meta/<schema>')
    def query_metadata(schema: str):
        if schema.startswith("m."):
            meta_info, _ = retrieve_ent_meta_info([schema])
            render_info = meta_info[list(meta_info.keys())[0]]
        else:
            meta_info, _ = retrieve_schema_meta_info([schema])
            render_info = meta_info[list(meta_info.keys())[0]]
        render_info["ent_id"] = schema
        return render_template('card.html',
                               **render_info)

    @app.route('/assets/<path:path>')
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        if static_dir is not None:
            return send_from_directory(os.path.join(static_dir, "assets"), path)

    return app


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_path,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    # Hack
    set_sparql_wrapper(args.sparql_endpoint_uri)

    predictor = Predictor.from_archive(archive, args.predictor)
    setattr(predictor, "world", args.world)
    setattr(predictor, "schema_retriever_uri", args.schema_retriever_uri)
    setattr(predictor, "sparql_endpoint_uri", args.sparql_endpoint_uri)

    return predictor


def main(args):
    # Executing this file with no extra options runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this, except possibly to test changes to the stock HTML).

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--archive-path', type=str, required=True, help='path to trained archive file')
    parser.add_argument('--world', type=str, required=True, help='the world want to demo with')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--test-case-path', type=str, help='path to test cases')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--static-dir', type=str, help='serve index.html from this directory')
    parser.add_argument('--field-name', type=str, action='append',
                        help='field names to include in the demo')
    parser.add_argument('--port', type=int, default=6000, help='port to serve the demo on')
    parser.add_argument('--schema_retriever_uri', type=str, default="http://localhost:6100/kbqa/api/v1.0/retriever",
                        help='URI for the schema retriever service')
    parser.add_argument('--sparql_endpoint_uri', type=str, default="http://localhost:8890/sparql/",
                        help='URI for the sparql endpoint')

    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')

    args = parser.parse_args(args)

    # Load modules
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    predictor = _get_predictor(args)

    field_names = args.field_name

    app = make_app(predictor=predictor,
                   field_names=field_names,
                   static_dir=args.static_dir)
    CORS(app)

    print('Using config: ' + json.dumps(vars(args)))

    print('Launching service on port: ' + str(args.port))

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print('Models loading...')
    http_server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
