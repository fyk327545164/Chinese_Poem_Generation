from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from model import Model


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():

    print("Building Vocab...")
    vocab = []
    with open('model/vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for word in lines:
            vocab.append(word)

    titles = [['春','意','浓'] for _ in range(2)]

    mode = 'inference'

    with tf.Graph().as_default():

        print("Building model...")

        model = Model(mode=mode, vocab_size=len(vocab))

        model.build_model()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            sess.run(tf.tables_initializer())

            saver = tf.train.Saver()
            print("Loading Checkpoint....")
            saver.restore(sess, tf.train.latest_checkpoint('./model/ckp/'))
            print("Checkpoint loaded at step " + str(model.global_step.eval()))

            sens = sess.run(model.output, feed_dict={model.title2sent_title_holder: titles})

            # print(sens)
            print(titles[0])
            for sen in sens:
                s = []
                # print(sen)
                for i in sen[0]:
                    s.append(vocab[i][:-1])
                print(s)

main()
