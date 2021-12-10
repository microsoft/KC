#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd "$RETRACK_HOME"/retriever/entitylinking/BERT-NER/ || exit

echo "Starting NER service..."
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 8009 2>&1 >/dev/null ); then
    echo "NER GrailQA online"
else
  python ner_api.py --model_path "$DATA_PATH/KBSchema/EntityLinking/NERModel/GrailQA" --port 8009 &
fi
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 8010 2>&1 >/dev/null ); then
    echo "NER WebQSP online"
else
  python ner_api.py --model_path "$DATA_PATH/KBSchema/EntityLinking/NERModel/WebQSP" --port 8010 &
fi
cd "$RETRACK_HOME"