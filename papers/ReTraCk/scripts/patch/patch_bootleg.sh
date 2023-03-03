#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd $RETRACK_HOME
cd ..
git clone https://github.com/HazyResearch/bootleg
cd bootleg
git checkout 37b0e67af8a790673a445798339db0d90f009e13

patch -p0 -l -i ../ReTraCk/retriever/diff/bootleg.diff

echo "Removing unnecessary assets..."

rm -rf .git
rm -rf configs
rm -rf data
rm -rf docs
rm -rf testcd
rm -rf tutorials
rm -f .gitignore
rm -f .travis.yml
rm -f *.md
rm -f download*.sh
#rm -f requirements.txt
#rm -f setup.py

echo "Copying assets to ReTraCk location"

/bin/cp -rf ./ ../ReTraCk/retriever/entitylinking/bootleg/

/bin/cp -rf ../ReTraCk/retriever/diff/bootleg/configs/ ../ReTraCk/retriever/entitylinking/bootleg/
/bin/cp -rf ../ReTraCk/retriever/diff/bootleg/*.py ../ReTraCk/retriever/entitylinking/bootleg/

echo "Cleanup..."

cd ..
rm -rf bootleg
cd $RETRACK_HOME