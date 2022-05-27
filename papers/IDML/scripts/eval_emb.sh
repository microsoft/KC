# evaluate protonet or protaugment, need specify the trained roberta-base model path as output-path
# dataset=Banking77
# for K in 1 5; do
#  for split in 1 2 3 4 5; do
#    CUDA_VISIBLE_DEVICES=0  python proto/proto_emb.py \
#      --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
#      --output-path output/${dataset}/0${split}/proto-5way${K}shot-robert-base-mlm-euc \
#      --encoder bert \
#      --model-name-or-path roberta-base	\
#      --load_ckpt True \
#      --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric euclidean --pooling avg
#  done
# done

# evaluate SBERT-para.
dataset=BANKING77
for K in 1 5; do
 for split in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python proto/proto_emb.py \
        --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
        --output-path output/${dataset}/0${split}/proto-5way${K}shot-paraphrase-distilroberta-base-v2-euc-wotrain/ \
        --encoder sentbert \
        --model-name-or-path paraphrase-distilroberta-base-v2 \
        --load_ckpt False \
        --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric euclidean --pooling avg
 done
done


# evaluate SBERT-NLI
dataset=BANKING77
for K in 1 5; do
 for split in 1 2 3 4 5; do
   CUDA_VISIBLE_DEVICES=0  python proto/proto_emb.py \
     --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
     --output-path output/${dataset}/0${split}/proto-5way${K}shot-nli-roberta-base-v2-euc-wotrain \
     --encoder sentbert \
     --model-name-or-path nli-roberta-base-v2	\
     --load_ckpt False \
     --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric euclidean --pooling avg
 done
done

# evaluate SimCSE-NLI
dataset=BANKING77
for K in 1 5; do
 for split in 1 2 3 4 5; do
   CUDA_VISIBLE_DEVICES=0  python proto/proto_emb.py \
     --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
     --output-path output/${dataset}/0${split}/proto-5way${K}shot-simcse-nli-euc-wotrain \
     --encoder bert \
     --model-name-or-path princeton-nlp/sup-simcse-roberta-base	\
     --load_ckpt False \
     --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric euclidean --pooling avg
 done
done

# evaluate DeCLUTR
dataset=BANKING77
for K in 1 5; do
 for split in 1 2 3 4 5; do
   CUDA_VISIBLE_DEVICES=0  python proto/proto_emb.py \
     --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
     --output-path output/${dataset}/0${split}/proto-5way${K}shot-declutr-base-euc-wotrain \
     --encoder bertmean \
     --model-name-or-path johngiorgi/declutr-base	\
     --load_ckpt False \
     --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric euclidean --pooling avg
 done
done

#evaluate SP-para., note that change the model_name_or_path to the downloaded model folder
dataset=BANKING77
for K in 1 5; do
 for split in 1 2 3 4 5; do
   CUDA_VISIBLE_DEVICES=0  python proto/proto_emb.py \
     --test-path data/${dataset}/few_shot/0${split}/test.jsonl \
     --output-path output/${dataset}/0${split}/proto-5way${K}shot-sp-paraphrase-cos-wotrain \
     --encoder para \
     --model-name-or-path paraphrase-at-scale-english \
     --load_ckpt False \
     --n-test-episodes 600 --n-support ${K} --n-classes 5 --n-query 5 --metric cosine --pooling avg
 done
done