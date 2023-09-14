---
license: mit
datasets:
- grail_qa
language:
- en
pipeline_tag: text2text-generation
---

This repo contains the **GrailQA target logical form generation** model of the EMNLP 2022 paper [TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Base](https://arxiv.org/abs/2210.12925).

[[code](https://github.com/microsoft/KC/tree/main/papers/TIARA)] [[poster](https://yihengshu.github.io/homepage/EMNLP22poster.pdf)] [[slides](https://yihengshu.github.io/homepage/EMNLP22slides.pdf)] [[video](https://s3.amazonaws.com/pf-user-files-01/u-59356/uploads/2022-11-04/fr03tjr/EMNLP22.mp4)] [[ACL Anthology](https://aclanthology.org/2022.emnlp-main.555/)] [[BibTeX](https://aclanthology.org/2022.emnlp-main.555.bib)]

## Usage

HuggingFace Transformer and Task: `T5ForConditionalGeneration`.

This model and the contexts of logical form and schema is a part of [TIARA_DATA.zip](https://kcpapers.blob.core.windows.net/tiara-emnlp2022/TIARA_DATA.zip). 

If you use it for TIARA, put it under the dir `<TIARA_root_dir>/model/grailqa_generation/lf_schema/`.

Input format: `question|query|<top-5 logical form>|entity|<entity label> <entity mid>|class|<top-10 class>|relation|<top-10 relation>`, e.g.,

```
what napa county wine is 13.9 percent alcohol by volume? |query|(AND wine.wine (JOIN wine.wine.percent_...|...|entity|napa valley m.0l2l_ |class|wine.wine|wine.wine_type|wine.vineyard|...|relation|wine.wine.percentage_alcohol|wine.wine_region|...
```

Output: a target logical form, e.g., 

```
(AND wine.wine (AND (JOIN (R wine.wine_sub_region.wines) m.0l2l_) (JOIN wine.wine.percentage_alcohol 13.9^^http://www.w3.org/2001/XMLSchema#float)))
```

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