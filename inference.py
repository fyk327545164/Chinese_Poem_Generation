from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from transformer import Transformer
from model import Model


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(title, model_mode='seq2seq'):

    print("Building Vocab...")
    vocab = []
    with open('model/vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for word in lines:
            vocab.append(word)

    titles = [title.split() for _ in range(2)]

    mode = 'inference'

    with tf.Graph().as_default():

        print("Building model...")

        if model_mode == 'seq2seq':
            model = Model(mode=mode, vocab_size=len(vocab))
            ckp_path = 'model/seq2seq_ckp/'
        else:
            model = Transformer(len(vocab), mode=mode)
            ckp_path = 'model/transformer_ckp/'

        model.build()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            sess.run(tf.tables_initializer())

            saver = tf.train.Saver()
            print("Loading Checkpoint....")
            saver.restore(sess, tf.train.latest_checkpoint(ckp_path))
            print("Checkpoint loaded at step " + str(model.global_step.eval()))

            sens = sess.run(model.out, feed_dict={model.title2sent_title_holder: titles})

            print(titles[0])
            for sen in sens:
                s = []
                for i in sen[0]:
                    s.append(vocab[i][:-1])
                print(s)


if __name__ == '__main__':
    main("秋意浓")
