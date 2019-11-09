from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import configuration
from tensorflow.python.layers import core as layers_core


class Model(object):

    def __init__(self, mode, vocab_size):

        # ['train', 'eval', 'inference']
        self.mode = mode

        self.config = configuration.Config()

        self.vocab_size = vocab_size + 1
        # Embedding Map
        self.embedding_map = tf.get_variable('embedding_map', [self.vocab_size, self.config.embedding_dim])

        self.vocab = tf.contrib.lookup.index_table_from_file('model/vocab.txt', num_oov_buckets=1)

        # title --> 1st sent
        self.title2sent_title_holder = tf.placeholder(tf.string, shape=[None, None])
        self.title2sent_sent_holder = tf.placeholder(tf.string, shape=[None, None])
        self.title2sent_sent_target_holder = tf.placeholder(tf.string, shape=[None, None])
        self.title2sent_title_length = tf.placeholder(tf.int32, shape=[None])
        # 1st sent --> 2nd sent
        self.sent2sent_sent1_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2sent_sent2_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2sent_title_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2sent_sent2_target_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2sent_title_length = tf.placeholder(tf.int32, shape=[None])
        # 1+2 sent --> 3rd sent
        self.sent2next_sent1_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2next_sent2_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2next_sent3_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2next_title_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2next_sent3_target_holder = tf.placeholder(tf.string, shape=[None, None])
        self.sent2next_title_length = tf.placeholder(tf.int32, shape=[None])

        self.title2sent_title = None
        self.title2sent_sent = None
        self.title2sent_sent_target = None

        self.sent2sent_sent1 = None
        self.sent2sent_sent2 = None
        self.sent2sent_title = None
        self.sent2sent_sent2_target = None

        self.sent2next_sent1 = None
        self.sent2next_sent2 = None
        self.sent2next_sent3 = None
        self.sent2next_title = None
        self.sent2next_sent3_target = None

        self.start_token = 0
        self.end_token = 1

        self.decoder_length = [6 for _ in range(self.config.batch_size)]

        if self.mode == 'inference':
            self.sent_length = [5 for _ in range(2)]
        else:
            self.sent_length = [5 for _ in range(self.config.batch_size)]

        self.output = None

        self.total_loss = None

        self.cross_entropy_loss = None

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.merge_summary = None

        self.eval_loss = tf.placeholder(tf.float32)

        self.perplexity = tf.placeholder(tf.float32)

        self.eval_summary = None

    def build_embedding(self):

        def embedding_helper(input_holder):
            return tf.nn.embedding_lookup(self.embedding_map, self.vocab.lookup(input_holder))

        if self.mode == 'inference':
            self.title2sent_title = embedding_helper(self.title2sent_title_holder)
            return

        self.title2sent_title = embedding_helper(self.title2sent_title_holder)
        self.title2sent_sent = embedding_helper(self.title2sent_sent_holder)
        self.title2sent_sent_target = self.vocab.lookup(self.title2sent_sent_target_holder)

        self.sent2sent_sent1 = embedding_helper(self.sent2sent_sent1_holder)
        self.sent2sent_sent2 = embedding_helper(self.sent2sent_sent2_holder)
        self.sent2sent_title = embedding_helper(self.sent2sent_title_holder)
        self.sent2sent_sent2_target = self.vocab.lookup(self.sent2sent_sent2_target_holder)

        self.sent2next_sent1 = embedding_helper(self.sent2next_sent1_holder)
        self.sent2next_sent2 = embedding_helper(self.sent2next_sent2_holder)
        self.sent2next_sent3 = embedding_helper(self.sent2next_sent3_holder)
        self.sent2next_title = embedding_helper(self.sent2next_title_holder)
        self.sent2next_sent3_target = self.vocab.lookup(self.sent2next_sent3_target_holder)

    def build_encoder(self, scope, input, dropout=True, seq_length=None, bidirection=False, trainable=True):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            if bidirection:

                fw_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units,
                                                               trainable=trainable)
                bw_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units,
                                                               trainable=trainable)

                if dropout:
                    fw_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(fw_encoder_cell,
                                                                    input_keep_prob=self.config.dropout_prob,
                                                                    output_keep_prob=self.config.dropout_prob)
                    bw_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(bw_encoder_cell,
                                                                    input_keep_prob=self.config.dropout_prob,
                                                                    output_keep_prob=self.config.dropout_prob)
                if seq_length is not None:
                    (fw_output, bw_output), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_encoder_cell,
                                                                                                   bw_encoder_cell,
                                                                                                   input,
                                                                                                   sequence_length=seq_length,
                                                                                                   dtype=tf.float32)
                else:
                    (fw_output, bw_output), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_encoder_cell,
                                                                                                   bw_encoder_cell,
                                                                                                   input,
                                                                                                   dtype=tf.float32)
                output = tf.concat([fw_output, bw_output], axis=-1)

                state_c = tf.concat((fw_state.c, bw_state.c), 1)
                state_h = tf.concat((fw_state.h, bw_state.h), 1)
                state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h)

            else:

                stacked_cell = []
                for _ in range(self.config.num_layer):

                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units, trainable=trainable)

                    if dropout:
                        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                                  input_keep_prob=self.config.dropout_prob,
                                                                  output_keep_prob=self.config.dropout_prob)
                    stacked_cell.append(lstm_cell)

                encoder_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_cell)

                if seq_length is not None:
                    output, state = tf.nn.dynamic_rnn(encoder_cell, input, sequence_length=seq_length, dtype=tf.float32)
                else:
                    output, state = tf.nn.dynamic_rnn(encoder_cell, input, dtype=tf.float32)

        return output, state

    def build_decoder(self, scope, encoder_output, encoder_state, input=None, seq_length=None, bidirection=False,
                      beam_search=False):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            if self.mode == 'inference':
                batch_size = 2
            else:
                batch_size = self.config.batch_size

            projection_layer = layers_core.Dense(self.vocab_size, use_bias=False)
            # if beam_search:
            #     encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.config.beam_width)
            #     encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.config.beam_width)
            #     # seq_length = tf.contrib.seq2seq.tile_batch(seq_length, multiplier=self.config.beam_width)

            attention_state = encoder_output
            seq_length = seq_length

            if bidirection:
                if scope == 'sent1_decoder':
                    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units * 2)
                else:
                    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units * 2)
                         for _ in range(self.config.num_layer)])

                if seq_length is not None:
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_lstm_units * 2,
                                                                            attention_state,
                                                                            memory_sequence_length=seq_length)
                else:
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_lstm_units * 2,
                                                                            attention_state)
            else:
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units)
                     for _ in range(self.config.num_layer)])

                if seq_length is not None:
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_lstm_units,
                                                                            attention_state,
                                                                            memory_sequence_length=seq_length)
                else:
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_lstm_units,
                                                                            attention_state)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.config.num_lstm_units // 2)

            # if beam_search: initial_state = decoder_cell.zero_state(batch_size * self.config.beam_width,
            # tf.float32).clone(cell_state=encoder_state) else:
            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            # if beam_search:
            #     decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
            #                                                    embedding=self.embedding_map,
            #                                                    start_tokens=tf.fill([batch_size], self.start_token),
            #                                                    end_token=self.end_token,
            #                                                    initial_state=initial_state,
            #                                                    beam_width=self.config.beam_width,
            #                                                    output_layer=projection_layer,
            #                                                    length_penalty_weight=0.0)
            # else:
            if self.mode == 'inference':
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_map,
                                                                  tf.fill([batch_size], self.start_token),
                                                                  self.end_token)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(input, self.decoder_length)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      initial_state,
                                                      output_layer=projection_layer)
            if self.mode == 'inference':

                output, state, seq_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=6)
                # impute_finished=True)
            else:
                output, state, seq_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

        return output, state, seq_lengths

    def build_model(self):

        self.build_embedding()

        if self.mode == 'inference':
            title_output, title_state = self.build_encoder('title_encoder', self.title2sent_title, dropout=False,
                                                           bidirection=True)

            sent1_out, _, _ = self.build_decoder('sent1_decoder', title_output, title_state,
                                                 bidirection=True)

            sent1 = sent1_out.sample_id
            # sent1 = sent1_output.predicted_ids
            sent1_embed = tf.nn.embedding_lookup(self.embedding_map, sent1)

            sent1_output, sent1_state = self.build_encoder("sent1_encoder", sent1_embed, dropout=False,
                                                           seq_length=self.sent_length)
            sent2_out, _, _ = self.build_decoder("sent2_decoder", sent1_output, sent1_state)

            sent2 = sent2_out.sample_id
            # sent2 = sent2_output.predicted_ids
            sent2_embed = tf.nn.embedding_lookup(self.embedding_map, sent2)

            _, sent2_state = self.build_encoder("sent2_encoder", sent2_embed, dropout=False,
                                                seq_length=self.sent_length)

            state_list = []
            for i in range(self.config.num_layer):
                state_c = tf.concat((sent1_state[i].c, sent2_state[i].c), 1)
                state_h = tf.concat((sent1_state[i].h, sent2_state[i].h), 1)
                state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h)
                state_list.append(state)
            sent2next_state = tuple(state_list)

            sent3_out, _, _ = self.build_decoder('sent3_decoder', title_output, sent2next_state, bidirection=True)

            sent3 = sent3_out.sample_id
            # sent3 = sent3_output.predicted_ids
            sent3_embed = tf.nn.embedding_lookup(self.embedding_map, sent3)

            sent3_output, sent3_state = self.build_encoder("sent1_encoder", sent3_embed, dropout=False,
                                                           seq_length=self.sent_length)
            sent4_out, _, _ = self.build_decoder("sent2_decoder", sent3_output, sent3_state)

            sent4 = sent4_out.sample_id
            # sent4 = sent4_output.predicted_ids

            self.output = [sent1, sent2, sent3, sent4]

        else:

            title2sent_output, title2sent_state = self.build_encoder('title_encoder', self.title2sent_title,
                                                                     seq_length=self.title2sent_title_length,
                                                                     bidirection=True)
            sent_out, _, _ = self.build_decoder('sent1_decoder', title2sent_output, title2sent_state,
                                                input=self.title2sent_sent,
                                                seq_length=self.title2sent_title_length,
                                                bidirection=True)
            logits_1 = sent_out.rnn_output

            # sent2sent_title_output, _ = self.build_encoder('title_encoder', self.sent2sent_title,
            #                                                seq_length=self.sent2sent_title_length)
            sent2sent_sent1_output, sent2sent_sent1_state = self.build_encoder('sent1_encoder', self.sent2sent_sent1)
            sent2_out, _, _ = self.build_decoder('sent2_decoder', sent2sent_sent1_output, sent2sent_sent1_state,
                                                 input=self.sent2sent_sent2)
            logits_2 = sent2_out.rnn_output

            sent2next_title_output, _ = self.build_encoder('title_encoder', self.sent2next_title,
                                                           seq_length=self.sent2next_title_length,
                                                           bidirection=True, trainable=False)
            _, sent2next_sent1_state = self.build_encoder('sent1_encoder', self.sent2next_sent1)
            _, sent2next_sent2_state = self.build_encoder('sent2_encoder', self.sent2next_sent2)

            state_list = []
            for i in range(self.config.num_layer):
                state_c = tf.concat((sent2next_sent1_state[i].c, sent2next_sent2_state[i].c), 1)
                state_h = tf.concat((sent2next_sent1_state[i].h, sent2next_sent2_state[i].h), 1)
                state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h)
                state_list.append(state)
            sent2next_state = tuple(state_list)

            sent3_out, _, _ = self.build_decoder('sent3_decoder', sent2next_title_output, sent2next_state,
                                                 input=self.sent2next_sent3,
                                                 bidirection=True,
                                                 seq_length=self.sent2next_title_length)

            logits_3 = sent3_out.rnn_output

            title2sent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.title2sent_sent_target,
                                                                             logits=logits_1)
            sent2sent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sent2sent_sent2_target,
                                                                            logits=logits_2)
            sent2next_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sent2next_sent3_target,
                                                                            logits=logits_3)

            self.cross_entropy_loss = [title2sent_losses, sent2sent_losses, sent2next_losses]

            title2sent_loss = tf.reduce_mean(title2sent_losses)
            sent2sent_loss = tf.reduce_mean(sent2sent_losses)
            sent2next_loss = tf.reduce_mean(sent2next_losses)

            self.total_loss = title2sent_loss + sent2sent_loss + sent2next_loss

            summary_loss = tf.summary.scalar("loss", self.total_loss)
            summary_title2sent_loss = tf.summary.scalar("title2sent_loss", title2sent_loss)
            summary_sent2sent_loss = tf.summary.scalar("sent2sent_loss", sent2sent_loss)
            summary_sent2next_loss = tf.summary.scalar("sent2next_loss", sent2next_loss)

            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.name, var)

            self.merge_summary = tf.summary.merge([summary_loss,
                                                   summary_title2sent_loss,
                                                   summary_sent2sent_loss,
                                                   summary_sent2next_loss])

            summary_eval_loss = tf.summary.scalar("eval_loss", self.eval_loss)
            summary_eval_perplexity = tf.summary.scalar("eval_perplexity", self.perplexity)

            self.eval_summary = tf.summary.merge([summary_eval_loss, summary_eval_perplexity])