# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# clone GrailQA entity linker
git clone https://github.com/dki-lab/GrailQA/
cd GrailQA
git checkout 27540d4
cd ..
mv GrailQA/allennlp .
mv GrailQA/entity_linker/* retriever/entity_linker/
rm -rf GrailQA

# download official GrailQA dataset (https://dki-lab.github.io/GrailQA/)
wget -c "https://dl.dropboxusercontent.com/s/3t9t1t14bh0563v/GrailQA_v1.0.zip?dl=1" -O GrailQA_v1.0.zip
unzip -o GrailQA_v1.0.zip
mv GrailQA_v1.0/* ../dataset/GrailQA
rm GrailQA_v1.0.zip
rm -rf GrailQA_v1.0

# download official WebQSP dataset (https://www.microsoft.com/en-us/download/details.aspx?id=52763)
wget -c "https://download.microsoft.com/download/F/5/0/F5012144-A4FB-4084-897F-CFDA99C60BDF/WebQSP.zip" -O WebQSP.zip
unzip -o WebQSP.zip
mv WebQSP/data/* ../dataset/WebQSP
rm WebQSP.zip
rm -rf WebQSP

# create directories for models
mkdir ../logs
mkdir ../model
mkdir ../model/grailqa_generation
mkdir ../model/grailqa_generation/lf_schema
mkdir ../model/grailqa_generation/lf
mkdir ../model/grailqa_generation/schema
mkdir ../model/grailqa_generation/none

mkdir ../model/webqsp_generation
mkdir ../model/webqsp_generation/lf_relation
mkdir ../model/webqsp_generation/lf
mkdir ../model/webqsp_generation/relation
mkdir ../model/webqsp_generation/none

mkdir ../model/schema_dense_retriever
mkdir ../model/schema_dense_retriever/class
mkdir ../model/schema_dense_retriever/relation

mkdir ../model/webqsp_schema_dense_retriever
mkdir ../dataset/GrailQA/schema_linking_results
mkdir ../dataset/GrailQA/schema_tokens/

# Entity mention data for RnG-KBQA GrailQA entity linker (https://github.com/salesforce/rng-kbqa)
# 1.3G, skip if you don't want to run entity linking
cd retriever/entity_linker/data
wget -c "https://7intqw.dm.files.1drv.com/y4mQBmOS8O8BeL2jPIrMdZWitr390Nc01VKU4grbRovRqfk2Jl7zzgjRAY9C-GSG9hE3T88LSgXCeRsg1zsWTWq3n1zqzo7Pxfh8-dxbO8tpFL-jYzupsqgr62d8Mg-wNVtwRTyV4c4t442KEIBxiBtNd1QzJAFukCjcpEYYsLXCEw8bn-hWl_IfR8w8-04rpcuGgseczimulJ5N2O-mnlB5w" -O mentions.zip
unzip -o mentions.zip
rm mentions.zip
cd ../../../

# Trained NER model for RnG-KBQA GrailQA entity linker (https://github.com/salesforce/rng-kbqa)
# 387M, skip if you don't want to run entity linking
cd retriever/entity_linker/BERT_NER
wget -c "https://7yntqw.dm.files.1drv.com/y4mXQQ4f_hKy5S2d4335XUPTi-4PgNaa0GFBKd2vYNqCMkVk4_yLNSzl-f29ZPX1BnZisBiVr7B_TRIHGe4Vic2UqY1Dtpqkr1lM84w5eje9yG1qDgDWoT15y1Wp43q7zjASDvfP_L0ixMPJVwN4PGZOR7G2NS6GP_9csge9oiWqDR_MUb-VyhBGMSIvYWOAoklJwU2HVBigFXH_QAU3YaFQA" -O trained_ner_model.zip
unzip -o trained_ner_model.zip
rm trained_ner_model.zip
cd ../../../

mkdir retriever/models
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/models/BertRanker.py" -O BertRanker.py
mv BertRanker.py retriever/models

wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/models/RobertaRanker.py" -O RobertaRanker.py
mv RobertaRanker.py retriever/models

wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/models/model_utils.py" -O model_utils.py
mv model_utils.py retriever/models

mkdir retriever/components
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/disamb_dataset.py" -O disamb_dataset.py
mv disamb_dataset.py retriever/components

wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/grail_utils.py" -O grail_utils.py
mv grail_utils.py retriever/components

mkdir retriever/executor
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/executor/sparql_executor.py" -O sparql_executor.py
mv sparql_executor.py retriever/executor

mkdir ../dataset/GrailQA/ontology
wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/ontology/domain_dict" -O domain_dict
mv domain_dict ../dataset/GrailQA/ontology

wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/ontology/domain_info" -O domain_info
mv domain_info ../dataset/GrailQA/ontology

wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/ontology/fb_roles" -O fb_roles
mv fb_roles ../dataset/GrailQA/ontology

wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/ontology/fb_types" -O fb_types
mv fb_types ../dataset/GrailQA/ontology

wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/ontology/reverse_properties" -O reverse_properties
mv reverse_properties ../dataset/GrailQA/ontology

wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/executor/logic_form_util.py" -O logic_form_util.py
mv logic_form_util.py retriever/executor

mkdir retriever/outputs
cp ../dataset/GrailQA/grailqa_v1.0_train.json retriever/outputs/grailqa_v1.0_train.json
cp ../dataset/GrailQA/grailqa_v1.0_dev.json retriever/outputs/grailqa_v1.0_dev.json
cp ../dataset/GrailQA/grailqa_v1.0_test_public.json retriever/outputs/grailqa_v1.0_test.json

mkdir retriever/feature_cache
mkdir retriever/misc
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/GrailQA/misc/relation_freq.json" -O relation_freq.json
mv relation_freq.json retriever/misc

mkdir ../dataset/WebQSP/entity_linking_results
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/misc/webqsp_train_elq-5_mid.json" -O ../dataset/WebQSP/entity_linking_results/webqsp_train_elq-5_mid.json
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/misc/webqsp_test_elq-5_mid.json" -O ../dataset/WebQSP/entity_linking_results/webqsp_test_elq-5_mid.json

mkdir ../dataset/WebQSP/RnG
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/parse_sparql.py" -O retriever/parse_sparql.py

mkdir cache
# Official cache of query results from KBs from RnG-KBQA - GrailQA (https://github.com/salesforce/rng-kbqa)
wget -c "https://storage.googleapis.com/sfr-rng-kbqa-data-research/KB_cache/grail.zip" -O cache/grail.zip
unzip -o cache/grail.zip -d cache
rm cache/grail.zip
mv cache/grail/* cache
rm -r cache/grail
# Official cache of query results from KBs from RnG-KBQA - WebQSP (https://github.com/salesforce/rng-kbqa)
wget -c "https://storage.googleapis.com/sfr-rng-kbqa-data-research/KB_cache/webqsp.zip" -O cache/webqsp.zip
unzip -o cache/webqsp.zip -d cache
mv cache/webqsp/* cache
rm cache/webqsp.zip
rm -rf cache/webqsp

wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/eval_topk_prediction.py" -O retriever/webqsp_eval_topk_prediction.py
wget -c "https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/enumerate_candidates.py" -O retriever/enumerate_candidates.py
cp utils/__init__.py retriever/executor/__init__.py

mkdir ../dataset/WebQSP/schema_linking_results
