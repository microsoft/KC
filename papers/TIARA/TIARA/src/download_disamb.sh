# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd retriever/checkpoints
wget https://storage.googleapis.com/sfr-rng-kbqa-data-research/model_release/grail_bert_entity_disamb.zip
unzip grail_bert_entity_disamb.zip
rm grail_bert_entity_disamb.zip
cd ../..