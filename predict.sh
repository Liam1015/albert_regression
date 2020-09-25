#!/usr/bin/env bash

export BERT_BASE_DIR=./albert_tiny
export GLUE_DIR=./data
export OUTPUT_DIR=./output

python run_regression.py \
  --task_name=scene \
  --do_predict=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt \
  --max_seq_length=32 \
  --output_dir=$OUTPUT_DIR/ \
  --predict_batch_size=128
