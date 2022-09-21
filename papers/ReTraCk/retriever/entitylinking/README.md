# Entity Linking for ReTraCk


### Bootleg Environment

#### Installation
```
conda create --name bootleg python=3.6
conda activate bootleg
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
cd bootleg/bootleg_aml
python setup.py develop
```

#### Training

1. NER
```
conda activate retrack_env
bash retriever/entitylinking/train_ner.sh GrailQA
```

```
conda activate retrack_env
bash retriever/entitylinking/train_ner.sh WebQSP
```
2. Bootleg
Please see retriever/entitylinking/bootleg/aml.ipynb on how to submit jobs on AML to train a Bootleg model. 
```shell
conda activate bootleg
python bootleg/bootleg_aml/run.py --config_script configs/grailqa_config.json --mode "train" --base_dir  --experiment_name --tensorboard_dir
```


#### Evaluation

1. Run NER model to extract entity mentions
```
conda activate retrack_env
cd ${RETRACK_HOME}
bash retriever/entitylinking/run_ner.sh GrailQA
bash retriever/entitylinking/run_ner.sh WebQSP
```

2. Run entity disambiguation and conduct evaluations

- Prior 
```
bash retriever/entitylinking/eval_prior.sh GrailQA
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| i.i.d.      | 79.40       | 77.90         | 78.64         | 1430        | 1403        | 1114          |
|compositional| 76.69       | 73.20         | 74.90         | 1321        | 1261        | 967           |
| zero-shot   | 70.99       | 68.39         | 69.67         | 3227        | 3109        | 2207          |
| all         | 74.28       | 71.73         | 72.98         | 5978        | 5773        | 4288          |

```
bash retriever/entitylinking/eval_prior.sh WebQSP
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| all         | 81.69       | 81.69         | 81.69         | 1846        | 1846        | 1508          |

Note: we fix a minor issue (excluding the NIL predictions) of the evaluation script for WebQSP dataset, this results in a small change in the precision metric. But this did not influence the conclusion.   

- Bootleg

```
conda activate bootleg
bash retriever/entitylinking/eval_bootleg.sh GrailQA
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| i.i.d.      | 80.47       | 78.95         | 79.70         | 1430        | 1403        | 1129          |
|compositional| 74.94       | 71.54         | 73.20         | 1321        | 1261        | 945           |
| zero-shot   | 63.17       | 60.86         | 61.99         | 3227        | 3109        | 1964          |
| all         | 69.95       | 67.55         | 68.73         | 5978        | 5773        | 4038          |

Note: Unfortunately, we did not store the checkpoint for the Bootleg model on the GrailQA dataset. We retrain the model, it got slightly better performance :). 

```
conda activate bootleg
bash retriever/entitylinking/eval_bootleg.sh WebQSP
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| all         | 58.33       | 58.61         | 58.47         | 1846        | 1855        | 1082          |


- Bootleg + Prior

```
conda activate bootleg
bash retriever/entitylinking/eval_bootleg.sh GrailQA
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| i.i.d.      | 87.17       | 85.52         | 86.34         | 1430        | 1403        | 1223          |
|compositional| 83.43       | 79.64         | 81.49         | 1321        | 1261        | 1052          |
| zero-shot   | 73.34       | 70.65         | 71.97         | 3227        | 3109        | 2280          |
| all         | 78.90       | 76.20         | 77.53         | 5978        | 5773        | 4555          |

Note: Unfortunately, we did not store the checkpoint for the Bootleg model on the GrailQA dataset. We retrain the model, it got slightly better performance :). 

```
conda activate bootleg
bash retriever/entitylinking/eval_bootleg.sh WebQSP
```
| Partition   | Precision   | Recall        | F1            | #GoldSupport| #PredSupport| #Correct      |
| :---        |    :----:   |          ---: |          ---: | :---        |    :----:   |          ---: |
| all         | 82.82       | 83.32         | 83.07         | 1846        | 1857        | 1538          |