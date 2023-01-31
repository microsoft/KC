# TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Base

This repository contains the open-sourced official implementation of the paper

[TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Base](https://arxiv.org/abs/2210.12925). Published at EMNLP 2022.

[[poster](./docs/EMNLP22poster.pdf)] [[slides](./docs/EMNLP22slides.pdf)] [[video](https://kcpapers.blob.core.windows.net/tiara-emnlp2022/EMNLP22.mp4)]

![TIARA.png](TIARA.png)

## Citation

If you find this paper or code useful, please cite the following paper:

```
@article{shu2022tiara,
  title={{TIARA}: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Bases},
  author={Shu, Yiheng and Yu, Zhiwei and Li, Yuhan and Karlsson, B{\"o}rje F. and Ma, Tingting and Qu, Yuzhong and Lin, Chin-Yew},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)},
  month = dec,
  year={2022}
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/2210.12925",
  doi = "",
}
```

## Requirements

The code is tested under the following environment:

- python>=3.8.13
- pytorch==1.11.0 (install cuda version if available): when install torch, make sure the cuda version is matched with your machine, and download the correct version
  from [here](https://download.pytorch.org/whl/torch_stable.html) is the best option; you can also check [pytorch.org](https://pytorch.org/get-started/locally/)
- other requirements: please see `requirements.txt`

## Datasets

- [GrailQA v1.0](https://dl.orangedox.com/WyaCpL/): train 44,337 / dev 6,763 / test 13,231 from [GrailQA leaderboard](https://dki-lab.github.io/GrailQA/)
- [WebQuestionsSP (WebQSP)](https://www.microsoft.com/en-us/download/details.aspx?id=52763): train 3,097 / test 1,638

## Setup

### 1. Setup Freebase

- Please follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso service.
  Note that at least 30G RAM and 53G+ disk space is needed for Freebase Virtuoso. The download may take some time. The default port of this service is `localhost:3001`.

### 2. Setup Conda Environment

Please modify the following command to select the appropriate [pytorch version](https://download.pytorch.org/whl/torch_stable.html) (for your machine's CUDA version and Python
version).

```shell
conda create --name tiara python=3.10
conda activate tiara
pip install -r requirements.txt

# an example of installing torch, you could also use other pytorch package or another way to install as described above
wget https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp310-cp310-linux_x86_64.whl 
pip install torch-1.11.0+cu113-cp310-cp310-linux_x86_64.whl
rm torch-1.11.0+cu113-cp310-cp310-linux_x86_64.whl
```

**[Optional]** If you would like to reinstall this environment, remove this environment first and then run the above command.

```shell
conda deactivate
conda env remove -n tiara
```

### 3. Setup Data and Code

TIARA requires some pre-processed data. It can be downloaded from [Azure Storage](https://kcpapers.blob.core.windows.net/tiara-emnlp2022/TIARA_DATA.zip) (7.8GB). Please download it and unzip the data folder before running.

```shell
wget
unzip TIARA_DATA.zip
```

**Note that** the following working directory is `src`.

```shell
cd src
python prepare.py
```

### 4. (Optional) GPU Adaptation

If the GPU takes up too much memory when running the following code, you may need to modify `chunk_size` in dense_retriever (for inference) or set gradient accumulation in the generator (for training) due to the difference in memory size supported by different GPUs. 

## Running

### 1. Entity Retrieval

**GrailQA** entity retrieval: mention detection (SpanMD) + candidate generation (FACC1) + entity disambiguation (BERT ranking);
**WebQSP** entity retrieval: [ELQ](https://github.com/facebookresearch/BLINK/tree/main/elq), results are stored in `dataset/WebQSP/entity_linking_results`.

The following is an introduction to GrailQA entity retrieval.

#### 1.1 Mention Detection

**Important notice:** To reproduce the entity linking results reported in the paper, please move the mention detection results to the correct location:

```shell
mv ../dataset/GrailQA/el_files retriever/
```

The mentions detected by our SpanMD model should be in `src/retriever/el_files/SpanMD_dev.json` and `src/retriever/el_files/SpanMD_test.json` for dev set and test set, respectively, and you can directly go to the step 1.2.

This is necessary as the code for SpanMD is not yet open-sourced. We provide a pre-trained mention detection model for other needs. If you want to train your own mention detector, we recommend you to utilize the provided mention detector model code based on [PURE](https://github.com/princeton-nlp/PURE), which can achieve comparable mention detection results to our SpanMD (88.94 F1 vs 86.07 F1). We provide the NER datasets and checkpoints in `../model/mention_detection/grailqa_data` and `../model/mention_detection/grailqa_models`, respectively. You can put these two folders under the PURE project and use the following scripts for training & inference.

```bash
# Run the NER model of PURE, the results will be stored in grailqa_models/checkpoint/ent_pred_test.json
python run_entity.py \
    --do_train --do_eval --eval_test \
    --learning_rate=5e-6 --task_learning_rate=5e-6 \
    --train_batch_size=32 \
    --eval_batch_size=108 \
    --context_window 0 \
    --max_span_length 15 \
    --task grailqa \
    --data_dir grailqa_data/json/ \
    --model bert-base-uncased \
    --output_dir grailqa_models/checkpoint \
    --num_epoch 10 \
    --seed 42

# Inference the NER model via checkpoints
python run_entity.py \
    --do_eval --eval_test \
    --context_window 0 \
    --task grailqa \
    --data_dir grailqa_data/json/ \
    --model allenai/scibert_scivocab_uncased \
    --output_dir grailqa_models/checkpoint
```

#### 1.2 Candidate Generation & Entity Disambiguation

If you skip this step, entity linking results are already in `../dataset/GrailQA/entity_linking_results`.

- Please run the following command, it will download
  checkpoints ([grail_bert_entity_disamb](https://storage.googleapis.com/sfr-rng-kbqa-data-research/model_release/grail_bert_entity_disamb.zip)) for entity disambiguation and put
  them under the `src/retriever/checkpoints/`
  folder:

```shell
sh download_disamb.sh
```

- Run the following command to get the final entity linking results for GrailQA dataset:

```shell
sh retriever/scripts/entity_retrieval.sh
```

The final entity linking results of this step can be found in `src/retriever/outputs`, and the disambiguation index can be found in `src/retriever/results/disamb`.

### 2. Exemplary Logical Form Retrieval

- Please refer to [RnG-KBQA](https://github.com/salesforce/rng-kbqa) for exemplary logical form retrieval including (iii) Enumerating Logical Form Candidates and (iv) Running
  Ranker.

### 3. Schema Retrieval

If no trained model exists in the model directory, i.e., `model/schema_dense_retriever` or `model/webqsp_schema_dense_retriever`, it will start with training and then run the
evaluation.

- Schema retrieval:

```shell
# GrailQA class and relation
sh grailqa_schema_retrieval.sh

# WebQSP relation
sh webqsp_schema_retrieval.sh
```

### 4. Target Logical Form Generation

The commands for both main and ablation experiments are as follows:

#### 4.1 GrailQA dev

```shell
# CD denotes constrained decoding, Schema denotes schema retrieval, ELF denotes exemplary logical form retrieval
# * denotes oracle entity annotations

# GrailQA dev
python algorithm/grailqa_generation.py --prompt lf_schema                 # TIARA
python algorithm/grailqa_generation.py --prompt lf_schema --checker False # TIARA w/o CD
python algorithm/grailqa_generation.py --prompt lf                        # TIARA w/o Schema
python algorithm/grailqa_generation.py --prompt lf --checker False        # TIARA w/o Schema & CD
python algorithm/grailqa_generation.py --prompt schema                    # TIARA w/o ELF
python algorithm/grailqa_generation.py --prompt schema --checker False    # TIARA w/o ELF & CD
python algorithm/grailqa_generation.py --prompt none                      # TIARA w/o ELF & schema
python algorithm/grailqa_generation.py --prompt none --checker False      # TIARA w/o ELF & schema & CD
python algorithm/grailqa_generation.py --prompt lf_schema --model_name t5-large
```

#### 4.2 GrailQA hidden test

```shell
# GrailQA hidden test
python algorithm/grailqa_generation.py --prompt lf_schema --run_valid False --run_test True
```

#### 4.3 WebQSP

```shell
# WebQSP
python algorithm/webqsp_generation.py --prompt lf_relation # TIARA
python algorithm/webqsp_generation.py --prompt lf          # TIARA w/o Schema
python algorithm/webqsp_generation.py --prompt relation    # TIARA w/o ELF
python algorithm/webqsp_generation.py --prompt none        # TIARA w/o ELF & Schema
```

#### 4.4 WebQSP with oracle entity

```shell
# WebQSP with oracle entity
python algorithm/webqsp_generation.py --golden_entity True --prompt lf_relation # TIARA*
python algorithm/webqsp_generation.py --golden_entity True --prompt lf # TIARA* w/o Schema
python algorithm/webqsp_generation.py --golden_entity True --prompt relation # TIARA* w/o ELF
python algorithm/webqsp_generation.py --golden_entity True --prompt none # TIARA* w/o ELF & Schema
```

## Model Training

If you want to train your own models, please see this section.

### 1. Train Schema Retriever

If a trained model exists in the model directory, i.e., `model/schema_dense_retriever` or `model/webqsp_schema_dense_retriever`, training will be skipped.

```shell
# `neg` denotes negative sampling, `num` denotes the number of negative samples
# `split` denotes the dataset to predict, not the dataset to train

# GrailQA class
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type class --split dev --neg random_question --num 20
# GrailQA relation
python retriever/dense_retriever/grailqa_schema_dense_retriever.py --schema_type relation --split dev --neg random_question --num 20

# WebQSP (relation)
python retriever/dense_retriever/webqsp_schema_dense_retriever.py --neg relation_sample --num 30

```

### 2. Train Logical Form Generator

If a trained model exists in the model directory, training will be skipped.

```shell
# GrailQA
python algorithm/grailqa_generation.py

# WebQSP
python algorithm/webqsp_generation.py
```

## Evaluation

### 1. Retrieval Evaluation

```shell
# Evaluation for GrailQA entity retrieval
python utils/statistics/entity_linking/grailqa_entity_statistics.py --data dev --el_path retriever/outputs/tiara_dev_el_results.json

# Evaluation for GrailQA schema retrieval
python retriever/schema_linker/grailqa_schema_evaluation.py

```

### 2. QA Evaluation with Official Scripts

QA prediction files are in the dir `logs`, please fill `<your_prediction_file_path>` below.

```shell
# GrailQA evaluation for dev set
python utils/statistics/grailqa_evaluate.py ../dataset/GrailQA/grailqa_v1.0_dev.json <your_prediction_file_path>

# WebQSP evaluation
python utils/statistics/webqsp_evaluate.py ../dataset/WebQSP/WebQSP.test.json <your_prediction_file_path>
```

For [GrailQA evaluation for hidden test set](https://worksheets.codalab.org/worksheets/0xd51b9aa5cf374ee598f1d6422cd976f3), please submit online.
