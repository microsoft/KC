# Issues with Entailment-based Zero-shot Text Classification

This repository contains the open-sourced official implementation of the paper

[Issues with Entailment-based Zero-shot Text Classification](https://aclanthology.org/2021.acl-short.99/) (ACL-IJCNLP 2021).  
_Tingting Ma, Jin-Ge Yao, Chin-Yew Lin, and Tiejun Zhao_

If you find this repo helpful, please cite the following paper

```tex
@inproceedings{ma-etal-2021-issues,
    title = {Issues with Entailment-based Zero-shot Text Classification},
    author = {Tingting Ma and Jin-Ge Yao and Chin-Yew Lin and Tiejun Zhao},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
    year = {2021},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2021.acl-short.99},
    pages = {786--796}
}
```

For any questions/comments, please feel free to open GitHub issues or email the <a href="mailto:hittingtingma@gmail.com">fist author<a/> directly.



## Requirements
    
The code requires  
```
python >= 3.6  
torch == 1.3.1  
transformers == 2.5.1  
nltk  
sklearn  
```
    
## Downloading datasets

For convenience, you can download the Yahoo, Emotion, and Situation datasets from [this link](https://drive.google.com/file/d/1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az/view). Only the test sets are utilized in the paper experiments. Note that we use the GLUE version validation set for SST-2.  

Download and preprocess other data by running the script below:

```
bash code/preprocess/download_data.sh
```

## Downloading pretrained model

For convenience, you can get the pretrained model used in Yin 2019 et al. at [this link](https://drive.google.com/file/d/1ILCQR_y-OSTdgkz45LP7JsHcelEsvoIn/view).

We also release the models we trained, which are used in Table 1 and Table 2. They can be downloaded by running thh script below:  

```
bash code/preprocess/download_model.sh
```

## Experiments

### 1. NSP  

To replicate our NSP (Reverse) results on the AGNews test set:  

Each input line has format:
```
text\tlabel\n
```
where input text and label are separated by a tab.

```python
python code/nsp/test_zero.py --input_fn data/test/agnews.txt \
    --label_fn data/template/agnews-nli.json \
    --output_dir experiments/agnews-nspr \
    --pretrain_model_dir bert-base-uncased \
    --use_nsp True \
    --reverse True \
    --label_single True
```

### 2. Variance experiment   

Run the test_zero.py script using the downloaded model:  

```python
python code/nsp/test_zero.py --input_fn data/test/agnews.txt \
    --label_fn data/template/agnews-nli.json \
    --output_dir experiments/agnews-mnli4 \
    --pretrain_model_dir experiments/model/mnli-4 \
    --use_nsp False \
    --reverse False \
    --label_single True
```

### 3. Shuffle the input word sequence    

use --random_input argument  

```python
python code/nsp/test_zero.py --input_fn data/test/agnews.txt \
    --label_fn data/template/agnews-nli.json \
    --output_dir experiments/agnews-nspr-rand \
    --pretrain_model_dir bert-base-uncased \
    --use_nsp True \
    --reverse True \
    --random_input True \
    --label_single True
```

### 4. Debias methods  

Note that we merge neutral and contradition classes into not-entailment class in all experiments. Also, we use four GPUs to train the NLI model.  

For the Debias-DA method, we directly merge the GLUE MNLI training dataset with [augmented data](https://github.com/Aatlantise/syntactic-augmentation-nli/blob/master/datasets/inv_trsf_large.tsv).

```python
python code/nsp/train_nli.py --data_dir data/nli/mnli-da \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name rte \
--do_train True \
--do_eval True \
--do_lower_case True \
--output_dir experiments/model/mnli-da \
--log_dir logs/mnli-da \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--per_gpu_train_batch_size 16 \
--seed 42 \
--max_seq_length 128
```

To train a model only using bias features:

```python
python code/debias/train_bias_only.py --input_dir data/nli/mnli \
--hans data/nli/hans \
--out_dir experiments/model/bias-only \
--w2v_fn data/crawl-300d-2M.vec
```

To train the Debias-Reweight model and evaluate it on Hans and MNLI matched dev sets:  

```python
python code/debias/train_debias.py --input_dir data/nli/mnli \
--hans_dir data/nli/hans \
--output_dir experiments/model/mnli-reweight \
--custom_bias experiments/model/bias-only/train.pkl \
--mode reweight_baseline \
--bert_model bert-base-uncased \
--do_train True \
--do_eval True \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--train_batch_size 64 \
--seed 42 \
--max_seq_length 128
```

For the Debias-BiasProduct:

```python
python code/debias/train_debias.py --input_dir data/nli/mnli \
--hans_dir data/nli/hans \
--output_dir experiments/model/mnli-biasproduct \
--custom_bias experiments/model/bias-only/train.pkl \
--mode bias_product_baseline \
--bert_model bert-base-uncased \
--do_train True \
--do_eval True \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--train_batch_size 64 \
--seed 42 \
--max_seq_length 128
```

## Note    
    
If you try to replicate the debias experiment results, the performance may be slightly different from the results presented in the paper.
This is due to randomness coming from pytorch training. The small expected variance doesn't impact our main conclusions.

## Acknowledgement

We thank the original authors for sharing their code, data or model:

[https://github.com/huggingface/transformers/tree/v2.5.1/examples](https://github.com/huggingface/transformers/tree/v2.5.1/examples)   
[https://github.com/yinwenpeng/BenchmarkingZeroShot](https://github.com/yinwenpeng/BenchmarkingZeroShot)    
[https://github.com/chrisc36/debias](https://github.com/chrisc36/debias)   
[https://github.com/UKPLab/emnlp2020-debiasing-unknown](https://github.com/UKPLab/emnlp2020-debiasing-unknown)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit httpscla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](httpsopensource.microsoft.comcodeofconduct).
For more information see the [Code of Conduct FAQ](httpsopensource.microsoft.comcodeofconductfaq) or
contact [opencode@microsoft.com](mailtoopencode@microsoft.com) with any additional questions or comments.
