# On the Effectiveness of Sentence Encoding for Intent Detection Meta-Learning

This repository contains the open-sourced official implementation of the paper

[On the Effectiveness of Sentence Encoding for Intent Detection Meta-Learning](https://openreview.net/pdf?id=SzGx4ZQfHZq) (NAACL 2022).   
_Tingting Ma, Qianhui Wu, Zhiwei Yu, Tie-jun Zhao and Chin-Yew Lin_

If you find this repo helpful, please cite the following paper
```tex
@inproceedings{ma-etal-2022-intentemb,
    title = {On the Effectiveness of Sentence Encoding for Intent Detection Meta-Learning},
    author = {Tingting Ma and Qianhui Wu and Zhiwei Yu and Tiejun Zhao and Chin-Yew Lin},
    booktitle = {2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2022)},
    year = {2022},
    publisher = {Association for Computational Linguistics},
    url = {https://openreview.net/pdf?id=SzGx4ZQfHZq}
}
```

For any questions/comments, please feel free to open GitHub issues or email the <a href="mailto:hittingtingma@gmail.com">fist author</a> directly.

## Requirements

```
pip install -r requirements.txt 
```
The original experiments were conducted with Python 3.7, numpy 1.21.0, and sentencepiece 0.1.94; which now have security vulneralibilities. Please try reverting to those versions in case of any problem.
  
## Download datasets  

For convenience, you can download the intent dataset splits from [this repo](https://github.com/tdopierre/ProtAugment/tree/main/data) and put the downloaded data into `data` folder.

## Download models

<!-- For sentence embedding models, we use public models as follows:

| sentence embeddings | download link |
| :-----:| :----: |
| SBERT-para. | [paraphrase-distilroberta-base-v2](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v2) |
| SBERT-NLI | [nli-roberta-base-v2](https://huggingface.co/sentence-transformers/nli-roberta-base-v2) |
| SimCSE-NLI | [princeton-nlp/sup-simcse-roberta-base](https://huggingface.co/princeton-nlp/sup-simcse-roberta-base) |
| DeCLUTR | [johngiorgi/declutr-base](https://github.com/JohnGiorgi/DeCLUTR) |
| SP-para. | [jwieting's repo](https://github.com/jwieting/paraphrastic-representations-at-scale) | -->

The SP-para. model need be downloaded manually by running

```
wget http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip
unzip paraphrase-at-scale-english.zip
```

## Experiments

### 1. Evaluate the sentence embedding  

```
bash scripts/eval_emb.sh
```

### 2. Apply the label name trick   

```
bash scripts/eval_emb_label.sh
```

### 3. Evaluate ProtoNet, ProtAugment  

Please following the [code](https://github.com/tdopierre/ProtAugment/) released by the author of ProtAugment, and 
save your trained model as encoder.pkl, then evaluate the model with our scripts by replacing the output_path argument.

## Acknowledgement  
This repository leverages open-sourced work from multiple sources. We thank the original authors of the repositories below for sharing their code, and models.

- [https://github.com/tdopierre/ProtAugment/](https://github.com/tdopierre/ProtAugment/);
- [https://github.com/jwieting/paraphrastic-representations-at-scale](https://github.com/jwieting/paraphrastic-representations-at-scale);
- [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers);
- [https://github.com/JohnGiorgi/DeCLUTR](https://github.com/JohnGiorgi/DeCLUTR);
- [https://github.com/princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)

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



  
