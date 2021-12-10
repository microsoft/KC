#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd $RETRACK_HOME
cd ..
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
git checkout 5fe254dd64d37332347edc73738edcb56096183f

patch -p0 -i ../ListQA/retriever/diff/scripts.diff
patch -p0 -i ../ListQA/retriever/diff/blink.diff
patch -p0 -i ../ListQA/retriever/diff/elq.diff

echo "Copying assets to ReTraCk location"

/bin/cp -rf blink/ ../ListQA/retriever/schema_retriever/dense_retriever/
/bin/cp -rf elq/ ../ListQA/retriever/schema_retriever/dense_retriever/
/bin/cp -rf elq_slurm_scripts/ ../ListQA/retriever/schema_retriever/dense_retriever/
/bin/cp -rf scripts/ ../ListQA/retriever/schema_retriever/dense_retriever/

/bin/cp -rf ../ListQA/retriever/diff/get_candidate_emb.py ../ListQA/retriever/schema_retriever/dense_retriever/blink/biencoder/get_candidate_emb.py
/bin/cp -rf ../ListQA/retriever/diff/evaluate_relation.py ../ListQA//retriever/schema_retriever/dense_retriever/elq/evaluate_relation.py

echo "Cleanup..."

cd ..
rm -rf BLINK
cd $RETRACK_HOME