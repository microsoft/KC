# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

freebase_addr = 'localhost'
freebase_port = '3001'

# datasets
input_path = '../'
output_path = '../'
grailqa_train_path = '../dataset/GrailQA/grailqa_v1.0_train.json'
grailqa_dev_path = '../dataset/GrailQA/grailqa_v1.0_dev.json'
grailqa_test_path = '../dataset/GrailQA/grailqa_v1.0_test_public.json'
grailqa_demo_path = '../dataset/GrailQA/grailqa_demo.json'
grailqa_entity_linking_path = '../dataset/GrailQA/entity_linking_results/grailqa_el.json'
grailqa_rng_el_path = '../dataset/GrailQA/entity_linking_results/rng_el_results/rng_el_results.json'  # including dev and test
grailqa_tiara_dev_el_path = '../dataset/GrailQA/entity_linking_results/tiara_dev_el_results.json'
grailqa_tiara_test_el_path = '../dataset/GrailQA/entity_linking_results/tiara_test_el_results.json'
webqsp_train_path = '../dataset/WebQSP/RnG/WebQSP.train.expr.json'
webqsp_test_path = '../dataset/WebQSP/RnG/WebQSP.test.expr.json'
webqsp_ptrain_path = '../dataset/WebQSP/RnG/WebQSP.ptrain.expr.json'
webqsp_pdev_path = '../dataset/WebQSP/RnG/WebQSP.pdev.expr.json'
webqsp_train_gen_path = '../dataset/WebQSP/RnG/webqsp_train_gen.json'
webqsp_test_gen_path = '../dataset/WebQSP/RnG/webqsp_test_gen.json'
webqsp_demo_path = '../dataset/webqsp_demo.json'
openke_dir_path = '../dataset/openke'

# model
grailqa_generation_model_path = '../model/grailqa_generation'
grailqa_domain_model_path = '../model/grailqa_domain_retrieval'
grailqa_relation_model_path = '../model/grailqa_relation_retrieval'
webqsp_generation_model_path = '../model/webqsp_generation'

# ontology
grailqa_class_name_dict_path = '../dataset/GrailQA/grailqa_class_name_dict.bin'
grailqa_mention_class_dict_path = '../dataset/GrailQA/grailqa_mention_class_dict.bin'
grailqa_reverse_properties_dict_path = '../dataset/GrailQA/ontology/reverse_properties'
grailqa_property_roles_dict_path = '../dataset/GrailQA/ontology/fb_roles'
grailqa_fb_types_path = '../dataset/GrailQA/ontology/fb_types'

# schema
grailqa_schema_train_path = '../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_train.jsonl'
grailqa_schema_dev_path = '../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev.jsonl'
grailqa_schema_test_path = '../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_test.jsonl'
class_token_path = '../dataset/GrailQA/schema_tokens/class_tokens.pkl'
relation_token_path = '../dataset/GrailQA/schema_tokens/relation_tokens.pkl'
class_trie_path = '../dataset/GrailQA/schema_tokens/class_trie.pkl'
relation_trie_path = '../dataset/GrailQA/schema_tokens/relation_trie.pkl'
webqsp_relation_trie_path = '../dataset/WebQSP/relation_trie.pkl'

webqsp_schema_ptrain_path = '../dataset/WebQSP/schema_linking_results/webqsp_train_relations.json'
webqsp_schema_pdev_path = '../dataset/WebQSP/schema_linking_results/webqsp_dev_relations.json'
webqsp_schema_test_path = '../dataset/WebQSP/schema_linking_results/webqsp_test_relations.json'

# logical form
grailqa_rng_ranking_train_path = '../dataset/GrailQA/RnG/train_rng_ranking_topk_results.json'
grailqa_rng_ranking_dev_path = '../dataset/GrailQA/RnG/dev_rng_ranking_topk_results.json'
grailqa_rng_ranking_test_path = '../dataset/GrailQA/RnG/test_rng_ranking_topk_results.json'
grailqa_tiara_ranking_dev_path = '../dataset/GrailQA/dev_tiara_ranking_topk_results.json'
grailqa_tiara_ranking_test_path = '../dataset/GrailQA/test_tiara_ranking_topk_results.json'
webqsp_lf_candidates_train_path = '../dataset/WebQSP/RnG/webqsp_train_candidates-ranking.json'
webqsp_lf_candidates_test_path = '../dataset/WebQSP/RnG/webqsp_test_candidates-ranking.json'
webqsp_rng_ranking_train_path = '../dataset/WebQSP/RnG/webqsp_train_rng_ranking_topk_results.json'
webqsp_rng_ranking_test_path = '../dataset/WebQSP/RnG/webqsp_test_rng_ranking_topk_results.json'
webqsp_rng_oracle_entity_ranking_test_path = '../dataset/WebQSP/RnG/webqsp_test_rng_oracle_entity_ranking_topk_results.json'

simple_questions_train_path = '../dataset/SimpleQuestions/annotated_fb_data_train.txt'
simple_questions_dev_path = '../dataset/SimpleQuestions/annotated_fb_data_valid.txt'
simple_questions_test_path = '../dataset/SimpleQuestions/annotated_fb_data_test.txt'

# cache
freebase_cache_dir = '../fb_cache'
webqsp_train_elq_path = '../dataset/WebQSP/entity_linking_results/webqsp_train_elq-5_mid.json'
webqsp_test_elq_path = '../dataset/WebQSP/entity_linking_results/webqsp_test_elq-5_mid.json'
webqsp_train_oracle_entity_path = '../dataset/WebQSP/entity_linking_results/webqsp_train_oracle_mid.json'
webqsp_test_oracle_entity_path = '../dataset/WebQSP/entity_linking_results/webqsp_test_oracle_mid.json'
webqsp_question_2hop_relation_path = '../dataset/WebQSP/webqsp_2hop_relations.pkl'
