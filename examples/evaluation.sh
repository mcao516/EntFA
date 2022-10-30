#!/bin/bash

python evaluation.py \
    --source_path $SCRATCH/summarization/XSum/fairseq_files/test.source.100 \
    --target_path $SCRATCH/summarization/XSum/fairseq_files/test.target.100 \
    --cmlm_model_path $SCRATCH/BART_models/xsum_cedar_cmlm \
    --data_name_or_path $SCRATCH/summarization/XSum/fairseq_files/xsum-bin \
    --mlm_path $SCRATCH/BART_models/bart.large \
    --knn_model_path $HOME/EntFA/examples/knn_classifier.pkl;