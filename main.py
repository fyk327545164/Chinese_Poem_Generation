from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocess import *
from model import Model
from transformer import Transformer
import configuration
import time
import os
import math


def main():
    data_files = build_data_files()

    print("Building input data...")
    title2sent, sent2sent, sent2next, vocab_size = build_input_data(data_files)

    np.random.shuffle(title2sent)
    np.random.shuffle(sent2sent)
    np.random.shuffle(sent2next)

    eval_title2sent = title2sent[:3200]
    eval_sent2sent = sent2sent[:3200]
    eval_sent2next = sent2next[:3200]

    train_title2sent = title2sent[3200:]
    train_sent2sent = sent2sent[3200:]
    train_sent2next = sent2next[3200:]

    config = configuration.Config()

    mode = 'train'

    with tf.Graph().as_default():

        print("Building model...")
        model = Model(mode=mode, vocab_size=vocab_size)

        model.build_model()

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(model.total_loss, global_step=model.global_step)

        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            writer = tf.summary.FileWriter("model/log/", sess.graph)

            if len(os.listdir("./model/ckp")) > 1:
                for file in os.listdir("./model/ckp"):
                    if file.endswith(".meta"):
                        loader = tf.train.import_meta_graph('model/ckp/' + file)
                loader.restore(sess, tf.train.latest_checkpoint('./model/ckp/'))

            print("Begin Training")
            while model.global_step.eval() <= config.num_steps:

                step = model.global_step.eval()
                start_time = time.time()

                if step % config.save_checkpoint_step == 0:
                    print("Saving model to checkpoint......")
                    saver.save(sess, "./model/ckp/model_ckp", global_step=model.global_step)
                    print("Finish saving the model at step " + str(step))

                feed_train, train_title2sent, train_sent2sent, train_sent2next = build_input_batch(train_title2sent,
                                                                                                   train_sent2sent,
                                                                                                   train_sent2next,
                                                                                                   config.batch_size,
                                                                                                   model)

                try:
                    sess.run(train_op, feed_dict=feed_train)
                    train_loss, summary = sess.run([model.total_loss, model.merge_summary], feed_dict=feed_train)
                    writer.add_summary(summary, step)
                except:
                    print("feed error")

                end_time = time.time()

                print("After " + str(step) + " steps: training loss is " + str(round(train_loss, 5)) + "---" +
                      str(round(end_time - start_time, 2)) + "s/step")

                if step % config.eval_inf_step == 0:
                    print("--------------------")

                    print("Running Evaluation...")
                    perplexity, loss = run_eval(eval_title2sent, eval_sent2sent, eval_sent2next, config.batch_size,
                                                model, sess)
                    print("After " + str(step) + " steps: evaluation loss is " + str(loss))

                    print("Perplexity is " + str(perplexity))

                    print("--------------------")

                    eval_summary = sess.run(model.eval_summary, feed_dict={model.perplexity: perplexity,
                                                                           model.eval_loss: loss})
                    writer.add_summary(eval_summary, step)

            writer.close()


def trian_eval_transformer():

    data_files = build_data_files()

    print("Building input data...")
    title2sent, sent2sent, sent2next, vocab_size = build_input_data(data_files)

    np.random.shuffle(title2sent)
    np.random.shuffle(sent2sent)
    np.random.shuffle(sent2next)

    train_title2sent = title2sent
    train_sent2sent = sent2sent
    train_sent2next = sent2next

    with tf.Graph().as_default():

        if True:
            print("Building model...")
            model = Transformer(vocab_size)
            model.build()

            train_op = tf.train.AdamOptimizer().minimize(model.loss)

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                step = 0

                print("Begin Training")
                while step <= 20000:

                    start_time = time.time()

                    feed_train, train_title2sent, train_sent2sent, train_sent2next = build_transformer_batch(
                        train_title2sent, train_sent2sent, train_sent2next, 64, model)

                    sess.run(train_op, feed_dict=feed_train)
                    loss = sess.run(model.loss, feed_dict=feed_train)

                    end_time = time.time()
                    print("After " + str(step) + " steps: training loss is " + str(round(loss, 5)) + "---" +
                          str(round(end_time - start_time, 2)) + "s/step")

                    step += 1


def run_eval(eval_title2sent, eval_sent2sent, eval_sent2next, batch_size, model, sess):
    losses = 0.
    weights = 0.
    eval_loss = 0.

    index = 0
    for _ in range(50):

        feed_eval, eval_title2sent, eval_sent2sent, eval_sent2next = build_input_batch(eval_title2sent,
                                                                                       eval_sent2sent,
                                                                                       eval_sent2next,
                                                                                       batch_size,
                                                                                       model)

        try:
            loss, eval_l = sess.run([model.cross_entropy_loss, model.total_loss], feed_dict=feed_eval)

            eval_loss += eval_l

            index += 1

            losses += np.sum([np.sum(loss[0]), np.sum(loss[1]), np.sum(loss[2])])

            weights += batch_size * 3 * 6

        except:
            print("feed error")

    perplexity = math.exp(losses / weights)

    eval_loss = eval_loss / index

    return perplexity, eval_loss


if __name__ == '__main__':
    # main()
    trian_eval_transformer()
