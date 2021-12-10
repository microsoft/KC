# ReTraCk: A Flexible and Efficient Framework for Knowledge Base Question Answering

This repository contains the open-sourced official implementation of the paper

[ReTraCk: A Flexible and Efficient Framework for Knowledge Base Question Answering](https://aclanthology.org/2021.acl-demo.39.pdf/) (ACL-IJCNLP 2021 - Demo).
_Shuang Chen*, Qian Liu*, Zhiwei Yu*, Chin-Yew Lin, Jian-Guang Lou and Feng Jiang_

If you find this repo helpful, please cite the following paper

```bib
@inproceedings{chen-etal-2021-retrack,
    title = "{R}e{T}ra{C}k: A Flexible and Efficient Framework for Knowledge Base Question Answering",
    author = "Chen, Shuang  and
      Liu, Qian  and
      Yu, Zhiwei  and
      Lin, Chin-Yew  and
      Lou, Jian-Guang  and
      Jiang, Feng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-demo.39",
    doi = "10.18653/v1/2021.acl-demo.39",
    pages = "325--336",
    abstract = "We present Retriever-Transducer-Checker (ReTraCk), a neural semantic parsing framework for large scale knowledge base question answering (KBQA). ReTraCk is designed as a modular framework to maintain high flexibility. It includes a retriever to retrieve relevant KB items efficiently, a transducer to generate logical form with syntax correctness guarantees and a checker to improve transduction procedure. ReTraCk is ranked at top1 overall performance on the GrailQA leaderboard and obtains highly competitive performance on the typical WebQuestionsSP benchmark. Our system can interact with users timely, demonstrating the efficiency of the proposed framework.",
}
```

For any questions/comments, please feel free to open GitHub issues.

## Installation


### Pre-setup

Before configuring the ReTraCk environment, we suggest you install Anaconda (miniconda should be enough) to create a virtual environment as described in the setup instructions below.
If you prefer to use another virtual environment system, please adapt the instructions. Contributions to this README to use them are also welcome.
 
Due to the data requirements to run the KB side of things, in order to run ReTraCk also requires installing some services. 

If you are using Windows, we suggest you use the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/)

### ReTraCk dependencies

After cloning this repo and going to the ReTraCk code folder:

```bash
conda create --name retrack_env python=3.6
conda activate retrack_env
pip install -r requirements.txt
```

If PyTorch fails to install, please run the command below. Replace "cu101" with the CUDA version suited to your environment.

```bash
pip install torch===1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

If your GPU doesn't support CUDA cc 3.7 or higher, you'll need to compile pytorch from source.


### Virtuoso Open-source Setup

Virtuoso is used as a SPARQL endpoint to Freebase. 
Please follow the setup steps described in this [repo](https://github.com/dki-lab/Freebase-Setup) to install Virtuoso Open-source 7.2.* and load their Freebase dump.


### Redis

We use Redis to store the anchor relations and entity meta information associated with each Freebase entity. 

Please follow the steps described in this [guideline](https://redis.io/topics/quickstart) to install the service.

```bash
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
make install
```

The ```make install``` is optional`, if you don't want Redis system-available.

## Configuring ReTraCk
 
### Downloading ReTraCk resources

#### Redis dump files

#### Model checkpoints

#### Datasets
We conduct the experiments on two datasets: [GrailQA](https://dl.orangedox.com/WyaCpL/) and [WebQuestionsSP] (https://www.microsoft.com/en-us/download/details.aspx?id=52763). Both datasets follow the same format of GrailQA. We appreciate Gu Yu, the author of [GrailQA](https://dki-lab.github.io/GrailQA/), who provides us with the pre-processed WebQuestionsSP.

### Demo Setup

0. Define environment variables for the Virtuoso Open-source installation, data files location, and ReTraCk's directory. 
   
    ```bash
    export RETRACK_HOME="{ReTraCk repo clone location}"
    export DATA_PATH="{path to the downloaded data}"
    export VIRTUOSO_PATH="{Virtuoso install directory}"
    ```

1. Setup dependencies and launch endpoints for relevant services (Redis, SPARQL, NER)
    ```bash
    bash $RETRACK_HOME/scripts/setup_dep.sh
    bash $RETRACK_HOME/scripts/redis.sh
    bash $RETRACK_HOME/scripts/virtuoso.sh
    bash $RETRACK_HOME/scripts/ner.sh
    ```

    If Virtuoso doesn't start, you can try:
    
    ```bash
    python virtuoso.py start 3001 -d ./virtuoso_db
    ```

2. Launch ReTraCk Retriever API services

    Once all dependencies are up, the retriever api can be launched via:

    ```bash
    python schema_service_api.py &
    python retriever_api.py &
    ```
   Keep in mind that the first run of the system may be slow as pretrained models will be downloaded and cached locally.

4. Run the demo parser of interest (e.g., GrailQA)
   
   ```bash
   cd parser 
   python demo.py --config_path ./retrack/configs/demo_grailqa.json
   ```    

## Evaluation

## Semantic Parser Guide

The following guideline assumes your current working directory is `parser`. If not, please change to it with `cd parser`.

### Evaluate & Debug your Parser

To evaluate a trained semantic parser, you could use `evaluate.py` to calculate results and collect error cases. Note that you should specify the variable `model_file` as the model file with the suffix `.tar.gz` which is packed by AllenNLP after training.

The parameters to the evaluation script are specified in a json file. We provide the configurations for GrailQA dev (as the test set is private) and WebQSP test.  
For example, for the GrailQA dev set you should run the following command:
```bash
python evaluate.py --config_path=../retrack/configs/parser_eval_grailqa_dev.json
```
For the best possible results, please enable the complete checker (`use_beam_check`, `use_virtual_forward`, `use_type_checking`, and `use_entity_anchor`.

- `fb_roles_file`, `fb_types_file` and `reverse_properties_file`: these optional files are used to calculate the official graph-based exact match on GrailQA, and you can download them from [here](https://github.com/dki-lab/GrailQA/tree/main/ontology). If these files are provided, the exact match metric represents the graph-based one, otherwise the vanilla exact match metric is used.

If you are evaluating on GrailQA's validation set (i.e., there are ground-truth logic forms to compare), the reported metric will be automatically aggregated as below:
```
avg_exact_match: 0.80, com_exact_match: 1.00, zero_exact_match: 0.50, loss: 0.00 ||: : EXAMPLES it [TIME,  ? s/it]
```

If you are evaluating on GrailQA's test set and preparing the submit files to GrailQA, the file with the suffix `_output.json` under the output folder will be useful, which is compatible with the official evaluation script of GrailQA.
Notably, there will also be a file with the suffix `.md`, which can be used to track details inside the model prediction.

## ReTraCk Retriever Guide

To evaluate ReTraCk we provide the configs and models presented in its paper.
Configs take the form of json files, and you can assign the target world/dataset as either "GrailQA" or "WebQSP".

To reproduce the schema retriever results, please edit /tests/debug/launch_schema_retriever.py to reference the dataset you want to evaluate.

Once it points to the right dataset, run:
```bash
python ./tests/debug/launch_schema_retriever.py
```
Inference will generate Class and Relation predictions, along with merged results, and print out Recall numbers for both predictions types.

### Use ReTraCkRetriever in your codebase

For examples on how to use ReTraCkRetriever in your code base, please check /tests/debug/launch_schema_retriever.py and [TRAIN.MD](TRAIN.MD). 

## License
The ReTrack Demo code is MIT licensed. See the [LICENSE](LICENSE) file for details.