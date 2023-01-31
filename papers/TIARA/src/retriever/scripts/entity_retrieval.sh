# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd retriever
export split=dev
python detect_entity_mention.py --split $split
mkdir results
sh scripts/run_disamb.sh predict checkpoints/grail_bert_entity_disamb $split
cd .. # back to src
python utils/disamb_to_entity.py
