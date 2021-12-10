# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import os
import redis
from paths import root_path
from retriever.configs import config_utils


class TestRedis(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(root_path, "./configs/retriever_config.json")
        config = config_utils.get_config(config_path)
        self.entity_meta_info_redis = redis.Redis(host=config['kb_store_host'], port=config['entity_meta_port'], db=0)
        self.in_anchor_redis = redis.Redis(host=config['kb_store_host'], port=config['in_relation_port'], db=0)
        self.out_anchor_redis = redis.Redis(host=config['kb_store_host'], port=config['out_relation_port'], db=0)

    def test(self):
        line = b'{"en_label": "Brett Hull Hockey \'95", "en_desc": null, "prominent_type": ["cvg.game_version"], "types": ["cvg.game_version", "common.topic"]}'
        assert self.entity_meta_info_redis.get('m.045zxrp') == line

        line = b'{"id": "m.045zxrp", "in_relations": ["cvg.cvg_platform.games_on_this_platform", "cvg.computer_videogame.versions", "cvg.cvg_publisher.game_versions_published", "type.type.instance", "cvg.cvg_developer.game_versions_developed"]}'
        assert self.in_anchor_redis.get('m.045zxrp') == line

        line = b'{"id": "m.045zxrp", "out_relations": ["cvg.game_version.release_date", "cvg.game_version.developer", "common.topic.notable_types", "type.object.type", "cvg.game_version.publisher", "common.topic.notable_for", "cvg.game_version.game", "kg.object_profile.prominent_type", "cvg.game_version.platform", "type"]}'
        assert self.out_anchor_redis.get('m.045zxrp') == line

        print('Success.')


if __name__ == '__main__':
    unittest.main()
