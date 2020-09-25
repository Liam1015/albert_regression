# coding:utf-8
# Restore model from pb file and do prediction

import sys
import codecs
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib import predictor
import tokenization
from deploy_config import scene_config
from datetime import datetime
from deploy_config import cls_scene_helpers as cls_scene_h


from pathlib import Path

# PROJECT_PATH = Path(__file__).absolute().parent
# sys.path.insert(0, str(PROJECT_PATH))


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("vocab_file", "albert_tiny/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")


class bertPredict(object):
    def __init__(self, pb_path, vocab_path):
        # pb_path = './model_pb'
        self.pb_path = pb_path
        subdirs = [x for x in Path(self.pb_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        # latest = str(pb_path)

        self.vocab_idx, self.idx_vocab = self._load_vocab(vocab_path)
        self.predict_fn = predictor.from_saved_model(latest)


    def predict(self, inputs, max_seq_length):

        receive_lists = self._process_input(inputs, max_seq_length)
        results = []
        for input_ids, input_mask, segment_ids, input_value in receive_lists:
            start = time.time()
            result = self.predict_fn(
                {'input_ids': input_ids,
                 'input_mask': input_mask,
                 'segment_ids': segment_ids,
                 'input_value': input_value}
            )
            end = time.time()
            print("cost time is %f" % (end - start) + " seconds")
            results.append(result)
        return results

    def _process_input(self, inputs, max_seq_length):
        
        data = []
        # if type(inputs) == str or type(inputs) == unicode:
        if type(inputs) == str:
            data.append(inputs)
        elif type(inputs) == list:
            data = data + inputs

        data_list=[]
        for input in data:
            tokenizer = tokenization.FullTokenizer(
                vocab_file=FLAGS.vocab_file, do_lower_case=True)
            tokens_text = tokenizer.tokenize(input)
            if len(tokens_text) > max_seq_length - 2:
                tokens_text = tokens_text[0:(max_seq_length - 2)]
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_text:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            input_value = [0.0]
            tuple_ele = ([input_ids], [input_mask], [segment_ids], [input_value])
            data_list.append(tuple_ele)
        return data_list




    def _load_vocab(self, vocab_path):
        with codecs.open(vocab_path, 'r', 'utf-8') as file:
            vocab_idx = {}
            idx_vocab = {}
            for idx, vocab in enumerate(file):
                vocab = vocab.strip()
                idx = int(idx)
                vocab_idx[vocab] = idx
                idx_vocab[idx] = vocab
        return vocab_idx, idx_vocab


if __name__ == '__main__':
    name_t_start = datetime.now()
    bert = bertPredict('./model_pb', 'albert_tiny/vocab.txt')
    inputs = ['我草你妈', '我好开心', '打开吸顶灯','什么狗血剧情']
    # inputs = '你给我记住我是你爸爸'

    results = bert.predict(inputs, max_seq_length=128)
    res = []
    for result in results:
        res.append(result['output'][0][0])

    # for item in res:
    #     item = item.tolist()
    #     top2_result, top1_label, top1_prob, top2_label, top2_prob = cls_scene_h.translate_readable_logit(
    #         item, 2, scene_config.Data_tup_scene)
    #     print("top2_result is :%s" % (top2_result))
    for i in range(len(res)):
        result = "context: {}".format(inputs[i]) + '\t' + "pred: {:6f}".format(res[i])
        print(result)
    name_t_end = datetime.now()
    print("SceneCLSCNN cost: %f seconds" % (name_t_end - name_t_start).total_seconds())




