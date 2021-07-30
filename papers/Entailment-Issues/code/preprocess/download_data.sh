mkdir -p data/origin
mkdir -p data/test
mkdir -p data/nli

echo "download agnews"
wget https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv -O data/origin/agnews_test.csv
python code/preprocess/convert_file.py --task agnews --input_fn data/origin/agnews_test.csv --output_dir data/test

echo "download sst"
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -P data/origin
unzip data/origin/SST-2.zip -d data/origin
python code/preprocess/convert_file.py --task sst --input_fn data/origin/SST-2/dev.tsv --output_dir data/test

echo "download snips"
wget https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/test/seq.in -O data/origin/snips_sent.txt
wget https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/test/label -O data/origin/snips_label.txt
paste data/origin/snips_sent.txt data/origin/snips_label.txt > data/origin/snips_test.txt
python code/preprocess/convert_file.py --task snips --input_fn data/origin/snips_test.txt --output_dir data/test

echo "download RTE"
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip -P data/origin
unzip data/origin/RTE.zip -d data/origin

echo "download MNLI , preprocess MNLI"
mkdir data/nli/mnli
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip -P ./data/origin
unzip data/origin/MNLI.zip -d data/origin
python code/preprocess/convert_file.py --task mnli --input_dir data/origin/MNLI --output_dir data/nli/mnli

echo "download CB, preprocess CB"
mkdir data/nli/cb
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip -P data/origin
unzip data/origin/CB.zip -d data/origin
python code/preprocess/convert_file.py --task cb --input_dir data/origin/CB --output_dir data/nli/cb

echo "data from paper https://www.aclweb.org/anthology/2020.acl-main.212/"
mkdir data/nli/mnli-da
wget https://raw.githubusercontent.com/Aatlantise/syntactic-augmentation-nli/master/datasets/inv_trsf_large.tsv -P data/origin/
python code/preprocess/convert_file.py --task mnli-da --mnli_fn data/nli/mnli/train.tsv --aug_fn data/origin/inv_trsf_large.tsv --aug_mnli_fn data/nli/mnli-da/train.tsv
cp data/nli/mnli/dev.tsv data/nli/mnli-da/dev.tsv

echo "download Hans"
mkdir data/nli/hans
wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt -P data/nli/hans

echo "download fasttext word embedding"
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -P data/origin
unzip data/origin/crawl-300d-2M.vec.zip -d data/
