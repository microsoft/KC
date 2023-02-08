---
license: mit
language:
- en
metrics:
- accuracy
pipeline_tag: text-classification
datasets:
- grail_qa
---


This repo contains the **GrailQA relation retrieval** model of the EMNLP 2022 paper [TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Base](https://arxiv.org/abs/2210.12925).

[[code](https://github.com/microsoft/KC/tree/main/papers/TIARA)] [[poster](https://yihengshu.github.io/homepage/EMNLP22poster.pdf)] [[slides](https://yihengshu.github.io/homepage/EMNLP22slides.pdf)] [[video](https://s3.amazonaws.com/pf-user-files-01/u-59356/uploads/2022-11-04/fr03tjr/EMNLP22.mp4)] [[ACL Anthology](https://aclanthology.org/2022.emnlp-main.555/)] [[BibTeX](https://aclanthology.org/2022.emnlp-main.555.bib)]


## Usage

HuggingFace Transformer and Task: `BertForSequenceClassification`.

This model is a part of [TIARA_DATA.zip](https://kcpapers.blob.core.windows.net/tiara-emnlp2022/TIARA_DATA.zip). 

If you use it for TIARA, put it under the dir `<TIARA_root_dir>/model/schema_dense_retrieval/relation/`.

Input format: `question [SEP] relation`, e.g.,

```

what napa county wine is 13.9 percent alcohol by volume? [SEP] wine.wine.percentage_alcohol

```

Output: a matching score.

## Citation

```
@inproceedings{shu-etal-2022-tiara,
    title = "{TIARA}: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Base",
    author = {Shu, Yiheng  and
      Yu, Zhiwei  and
      Li, Yuhan  and
      Karlsson, B{\"o}rje  and
      Ma, Tingting  and
      Qu, Yuzhong  and
      Lin, Chin-Yew},
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.555",
}
```