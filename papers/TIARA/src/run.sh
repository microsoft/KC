# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# CD denotes constrained decoding, Schema denotes schema retrieval, ELF denotes exemplary logical form retrieval
# * denotes using oracle entity annotations

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

# GrailQA test
python algorithm/grailqa_generation.py --prompt lf_schema --run_valid False --run_test True

# WebQSP
python algorithm/webqsp_generation.py --prompt lf_relation # TIARA
python algorithm/webqsp_generation.py --prompt lf          # TIARA w/o Schema
python algorithm/webqsp_generation.py --prompt relation    # TIARA w/o ELF
python algorithm/webqsp_generation.py --prompt none        # TIARA w/o ELF & Schema

# WebQSP oracle entity annotations
python algorithm/webqsp_generation.py --golden_entity True --prompt lf_relation # TIARA*
python algorithm/webqsp_generation.py --golden_entity True --prompt lf          # TIARA* w/o Schema
python algorithm/webqsp_generation.py --golden_entity True --prompt relation    # TIARA* w/o ELF
python algorithm/webqsp_generation.py --golden_entity True --prompt none        # TIARA* w/o ELF & Schema

# Schema retrieval
python retriever/dense_retriever/webqsp_schema_dense_retriever.py --neg question_sample --num 20
