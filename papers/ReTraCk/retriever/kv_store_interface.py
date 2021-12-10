# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict
import pickle
import json
import os
import redis
from retriever.entitylinking.entity_linker import EntityLinker
from retriever.schema_retriever.client import DenseSchemaRetrieverClient
from retriever import utils
from retriever.configs import config_utils


class KBRetriever(object):

    def __init__(self, config):

        if not isinstance(config, Dict):
            config = config_utils.get_config(config)

        self.config = config
        self.entity_linker = EntityLinker(config=config)
        self.schema_retriever_client = DenseSchemaRetrieverClient(config)

        print("---------------------")
        print("Used KBRetriever configuration:\n", self.config)
        print("---------------------")

        kb_store = config['kb_store_host']
        self.entity_meta_info_redis = redis.Redis(host=kb_store, port=config['entity_meta_port'], db=0)
        self.in_anchor_redis = redis.Redis(host=kb_store, port=config['in_relation_port'], db=0)
        self.out_anchor_redis = redis.Redis(host=kb_store, port=config['out_relation_port'], db=0)

        # Test Redis is up
        self.test_redis_instances()

        self.schema_meta_info = self.load_pickle_file(os.path.join(config["base_data_dir"], config["schema_meta_info_fn"]))

    def test_redis_instances(self):
        try:
            val = self.entity_meta_info_redis.get("")
            val = self.in_anchor_redis.get("")
            val = self.out_anchor_redis.get("")
        except (ConnectionError, redis.exceptions.ConnectionError) as e:
            print(f'>>> Redis instance not up')
            raise EnvironmentError(f'Redis instances not up. {str(e)}')

    @staticmethod
    def load_pickle_file(pkl_fn):
        with open(pkl_fn, mode="rb") as fp:
            obj = pickle.load(fp)
            return obj

    def get_entity_meta_info(self, entity_id):
        response = self.entity_meta_info_redis.get(entity_id)
        if response:
            return json.loads(response)
        else:
            return {}

    def pack_one_node(self, nid, node_type, id, score=0.0, offset=None):

        if node_type == "entity":
            meta_info = self.get_entity_meta_info(id)
        else:
            meta_info = self.schema_meta_info.get(id, {})

        friendly_name = meta_info.get("en_label", "NIL")
        cls = meta_info.get("prominent_type", [])

        if len(cls) > 0:
            cls = cls[0]
        else:
            cls = ""

        classes = set(meta_info.get("types", {id}))
        if cls != "":
            classes = [cls] + list(classes - {cls})
        else:
            classes = list(classes)

        return {
            "nid": nid,
            "node_type": node_type,
            "id": id,
            "class": cls,
            "score": score,
            "offset": offset,
            "classes": classes,
            "friendly_name": friendly_name,
            "question_node": 0,
            "function": 'none'
        }

    def pack_dense_embedding_query_graph(self, entity_list, classes, relations, node_id=0):

        nodes = []
        edges = []

        for entity in entity_list:
            node = self.pack_one_node(node_id, "entity", entity.ent_id, entity.score, entity.offset)
            node_id += 1
            nodes.append(node)

        for cls in classes:
            node = self.pack_one_node(node_id, "class", cls[0], cls[1])
            nodes.append(node)
            node_id += 1

        for rel in relations:
            edge = self.pack_edge(-1, -1, rel[0], rel[1])
            edges.append(edge)

        return nodes, edges

    def pack_edge(self, start, end, relation, score):
        return {
            "start": start,
            "end": end,
            "relation": relation,
            "score": score,
            "friendly_name": self.get_relation_name(relation)
        }

    def get_relation_name(self, relation):
        if relation in self.schema_meta_info and self.schema_meta_info[relation]["en_label"]:
            return self.schema_meta_info[relation]["en_label"]

        relation_name = relation.split('.')[-1].split("_")
        return " ".join(["{}{}".format(word[0].upper(), word[1:]) for word in relation_name])

    def is_entity(self, s):
        if s.startswith('m.') or s.startswith('g.'):
            return True

    def get_in_relations(self, entity: str):
        response = self.in_anchor_redis.get(entity)
        in_relations = []
        if response:
            in_relations = json.loads(response)['in_relations']
        return in_relations

    def get_out_relations(self, entity: str):
        response = self.out_anchor_redis.get(entity)
        out_relations = []
        if response:
            out_relations = json.loads(response)['out_relations']
        return out_relations

    def gen_anchor_relations(self, entity: str):
        in_relations = self.get_in_relations(entity)
        out_relations = self.get_out_relations(entity)
        return in_relations, out_relations

    def predict(self, sentence, world=None):

        if world is None:
            world = self.config['world']

        sentence = sentence.lower()
        el_output = self.entity_linker.predict(sentence, topk=self.config['topk'])

        literal_nodes, node_id = utils.gen_all_literal_nodes(el_output, world=world)

        entity_list = utils.get_prior_el_topk(el_output)

        types, relations = self.schema_retriever_client.predict(sentence, world=world)

        nodes, edges = self.pack_dense_embedding_query_graph(entity_list, types, relations, node_id=node_id)

        in_out_relations = {}
        for ent in entity_list:
            in_relation, out_relation = self.gen_anchor_relations(ent.ent_id)
            in_out_relations[ent.ent_id] = {
                "in_relation": list(in_relation),
                "out_relation": list(out_relation)
            }

        output = {}

        output["bert_tokens"] = el_output["tokens"]
        output["graph_query"] = {
            "nodes": literal_nodes + nodes,
            "edges": edges
        }

        output["anchor_relations"] = in_out_relations
        output["entity_meta_info"] = {}

        for ent in entity_list:

            meta_info = self.get_entity_meta_info(ent.ent_id)
            en_label = meta_info.get("en_label", "NIL")

            if len(meta_info.get("prominent_type", [])) > 0:
                ent_type = meta_info["prominent_type"][0]
            else:
                ent_type = ""

            ent_desc = meta_info.get("en_desc", "")
            output["entity_meta_info"][ent.ent_id] = {
                "ent_name": en_label,
                "ent_type": ent_type,
                "ent_desc": ent_desc
            }

        return output
