# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

python retriever/schema_linker/webqsp_anchor_relations.py
python retriever/dense_retriever/webqsp_schema_dense_retriever.py --split dev
python retriever/dense_retriever/webqsp_schema_dense_retriever.py --split test
python retriever/dense_retriever/webqsp_schema_dense_retriever.py --split train # to train generator
