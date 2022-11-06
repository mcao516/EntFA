#!/bin/bash
module load cuda/11.0
module load python/3.8

source $HOME/envFS/bin/activate


SOURCE_PATH=$SCRATCH/summarization/XSum/fairseq_files/test.source
TARGET_PATH=$SCRATCH/summarization/XSum/fairseq_files/test.target

python evaluation.py \
    --source_path $SOURCE_PATH \
    --target_path $TARGET_PATH \
    --cmlm_model_path $SCRATCH/BART_models/xsum_cedar_cmlm \
    --data_name_or_path $SCRATCH/summarization/XSum/fairseq_files/xsum-bin \
    --mlm_path $SCRATCH/BART_models/bart.large \
    --knn_model_path $HOME/EntFA/examples/knn_classifier.pkl;