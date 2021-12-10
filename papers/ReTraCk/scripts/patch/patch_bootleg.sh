#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd $RETRACK_HOME
cd ..
git clone https://github.com/HazyResearch/bootleg
cd bootleg
git checkout d43ce8c9d775c6d20b8319087808014a9635605c

patch -p0 -l -i ../ListQA/retriever/diff/bootleg.diff

echo "Removing unnecessary assets..."

rm -rf .git
rm -rf configs
rm -rf data
rm -rf docs
rm -rf test
rm -rf tutorials
rm -f .gitignore 
rm -f .travis.yml 
rm -f *.md 
rm -f download*.sh
rm -f requirements.txt
rm -f setup.py

echo "Copying assets to ReTraCk location"

/bin/cp -rf ./ ../ListQA/retriever/entitylinking/bootleg/bootleg_aml

/bin/cp -rf ../ListQA/retriever/diff/bootleg/configs/ ../ListQA/retriever/entitylinking/bootleg/bootleg_aml
/bin/cp -rf ../ListQA/retriever/diff/bootleg/*.py ../ListQA/retriever/entitylinking/bootleg/bootleg_aml

echo "Cleanup..."

cd ..
rm -rf bootleg
cd $RETRACK_HOME