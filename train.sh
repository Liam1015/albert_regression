#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/albert_zh/albert_base_zh
#export GLUE_DIR=/home/home/zhaoxj/pycharmProjects/albert_zh/data
#export OUTPUT_DIR=/home/zhaoxj/pycharmProjects/albert_zh/output

export BERT_BASE_DIR=./albert_tiny
export GLUE_DIR=./data
export OUTPUT_DIR=./output


python run_regression.py \
  --task_name=scene \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=256 \
  --learning_rate=5e-5 \
  --num_train_epochs=150 \
  --output_dir=$OUTPUT_DIR/
