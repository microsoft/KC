#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

echo "Setup BERT-NER..."

cd "$RETRACK_HOME"/retriever/entitylinking/ || exit

DIR="./BERT-NER"
if [ -d "$DIR" ]; then
    echo "Code already checked out"
else
    git clone https://github.com/kamalkraj/BERT-NER.git
fi
cp ner_api.py BERT-NER/

cd "$RETRACK_HOME"

echo "Setup BLINK..."

cd "$RETRACK_HOME"

bash ./scripts/patch/patch_blink.sh

cd "$RETRACK_HOME"

echo "Setup bootleg..."

cd "$RETRACK_HOME"

bash ./scripts/patch/patch_bootleg.sh

cd "$RETRACK_HOME"