# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Arguments:
# --schema_type: class/relation
# --split: train/dev/test

# GrailQA dev class
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type class --split dev
# GrailQA dev relation
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type relation --split dev
# Merge dev class and relation as dev schema
python retriever/schema_linker/merge_class_relation.py --class_path ../logs/grailqa_dev_class_predictions.json --relation_path ../logs/grailqa_dev_relation_predictions.json --output_path ../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_dev.jsonl

# GrailQA train class
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type class --split train
# GrailQA train relation
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type relation --split train
# Merge train class and relation as train schema
python retriever/schema_linker/merge_class_relation.py --class_path ../logs/grailqa_train_class_predictions.json --relation_path ../logs/grailqa_train_relation_predictions.json --output_path ../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_train.jsonl

# Grail test class
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type class --split test
# Grail test relation
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type relation --split test
# Merge test class and relation as test schema
python retriever/schema_linker/merge_class_relation.py --class_path ../logs/grailqa_test_class_predictions.json --relation_path ../logs/grailqa_test_relation_predictions.json --output_path ../dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_test.jsonl
