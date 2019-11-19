import tensorflow as tf
import numpy as np
import math


class Configuration:
    def __init__(self):
        self.embedding_dim = 512

        self.multi_head_h = 8
        self.num_layer = 2

        self.k_dim = self.embedding_dim // self.multi_head_h
        self.v_dim = self.embedding_dim // self.multi_head_h

        self.batch_size = 64


class Transformer:
    def __init__(self, vocab_size, config=Configuration(), mode='train'):

        self.config = config

        self.vocab_size = vocab_size + 1

        self.mode = mode

        self.vocab = tf.contrib.lookup.index_table_from_file('model/vocab.txt', num_oov_buckets=1)

        self.title2sent_title_holder = tf.placeholder(tf.string, shape=[None, None], name='title2sent_title_holder')
        self.title2sent_sent_holder = tf.placeholder(tf.string, shape=[None, None], name='title2sent_sent_holder')
        self.title2sent_sent_target_holder = tf.placeholder(tf.string, shape=[None, None], name='title2sent_sent_target_holder')

        self.title2sent_input_mask_holder = tf.placeholder(tf.int32, shape=[None], name='title2sent_input_mask_holder')

        self.sent2sent_sent1_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2sent_sent1_holder')
        self.sent2sent_sent2_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2sent_sent2_holder')
        self.sent2sent_sent2_target_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2sent_sent2_target_holder')

        self.sent2next_sent_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2next_sent_holder')
        self.sent2next_sent3_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2next_sent3_holder')
        self.sent2next_sent3_target_holder = tf.placeholder(tf.string, shape=[None, None], name='sent2next_sent3_target_holder')

        self.sent2next_input_mask_holder = tf.placeholder(tf.int32, shape=[None], name='sent2next_input_mask_holder')

        self.title2sent_title = None
        self.title2sent_sent = None
        self.title2sent_sent_target = None
        self.title2sent_input_mask = None

        self.sent2sent_sent1 = None
        self.sent2sent_sent2 = None
        self.sent2sent_sent2_target = None

        self.sent2next_sent = None
        self.sent2next_sent3 = None
        self.sent2next_sent3_target = None
        self.sent2next_input_mask = None

        self.embedding_map = tf.get_variable('embedding_map', shape=[self.vocab_size, self.config.embedding_dim])

        self.out = None

        self.cross_entropy_loss = None

        self.total_loss = None

        self.merge_summary = None

        self.eval_loss = tf.placeholder(tf.float32)

        self.perplexity = tf.placeholder(tf.float32)

        self.eval_summary = None

        def build_positional_encoding(length):
            encoding = np.array([
                [pos / np.power(10000, 2 * (i // 2) / self.config.embedding_dim) for i in
                 range(self.config.embedding_dim)]
                if pos != 0 else np.zeros(self.config.embedding_dim) for pos in range(length)])

            encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
            encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])
            return tf.convert_to_tensor(np.array([encoding for _ in range(self.config.batch_size)]), dtype=tf.float32)

        self.positional_encoding_20 = build_positional_encoding(20)
        self.positional_encoding_10 = build_positional_encoding(10)
        self.positional_encoding_6 = build_positional_encoding(6)
        self.positional_encoding_5 = build_positional_encoding(5)

        self.inference_input_holder = tf.placeholder(tf.string, shape=[None, None])
        self.inference_input_length = tf.placeholder(tf.int32)
        # self.inference_positional_encoding = build_positional_encoding(self.inference_input_length)
        self.inference_input = None

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def build_embedding(self):

        def embedding_helper(x, x_encoding):
            x_encoding = x_encoding * (self.config.embedding_dim ** 0.5)
            return tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_map, self.vocab.lookup(x)) + x_encoding,
                                 keep_prob=0.9)

        def masking_helper(lengths, max_length):

            return tf.tile(
                tf.expand_dims(
                    tf.sequence_mask(lengths, max_length, dtype=tf.float32), axis=-1), [1, 1, self.config.embedding_dim])

        if self.mode == 'inference':
            self.inference_input = tf.nn.embedding_lookup(self.embedding_map,
                                                          self.vocab.lookup(self.title2sent_title_holder))
            return

        self.title2sent_title = embedding_helper(self.title2sent_title_holder, self.positional_encoding_10)
        self.title2sent_sent = embedding_helper(self.title2sent_sent_holder, self.positional_encoding_6)
        self.title2sent_sent_target = self.vocab.lookup(self.title2sent_sent_target_holder)
        self.title2sent_input_mask = masking_helper(self.title2sent_input_mask_holder, 10)

        self.sent2sent_sent1 = embedding_helper(self.sent2sent_sent1_holder, self.positional_encoding_5)
        self.sent2sent_sent2 = embedding_helper(self.sent2sent_sent2_holder, self.positional_encoding_6)
        self.sent2sent_sent2_target = self.vocab.lookup(self.sent2sent_sent2_target_holder)

        self.sent2next_sent = embedding_helper(self.sent2next_sent_holder, self.positional_encoding_20)
        self.sent2next_sent3 = embedding_helper(self.sent2next_sent3_holder, self.positional_encoding_6)
        self.sent2next_sent3_target = self.vocab.lookup(self.sent2next_sent3_target_holder)
        self.sent2next_input_mask = masking_helper(self.sent2next_input_mask_holder, 20)

    def build_encoder_layer(self, mode, inputs, inputs_mask=None):

        if inputs_mask is not None:
            out = tf.multiply(inputs, inputs_mask)
        else:
            out = inputs

        for num in range(self.config.num_layer):

            out = self.build_encoder_subLayer(out, mode + "_encoder_layer_" + str(num))

        return out

    def build_encoder_subLayer(self, inputs, layer_scope):

        multiHead_out = self.build_multiHead_attention(layer_scope, inputs, inputs, inputs)

        out = tf.math.l2_normalize(inputs + multiHead_out, axis=-1)

        FF_out = self.build_FFNN(layer_scope, out)

        return out + FF_out

    def build_decoder_layer(self, encoder_output, mode, inputs, mask_head=True):

        out = inputs

        for num in range(self.config.num_layer):

            out = self.build_decoder_sublayer(out, encoder_output, mode + "_decoder_layer_" + str(num), mask_head=mask_head)

        return out

    def build_decoder_sublayer(self, inputs, encoder_output, layer_scope, mask_head=True):

        if mask_head:
            masked_multiHead_out = self.build_multiHead_attention(layer_scope + "_mask", inputs, inputs, inputs, mask=True)

            out = tf.math.l2_normalize(inputs + masked_multiHead_out, axis=-1)

        else:
            out = inputs

        multiHead_out = self.build_multiHead_attention(layer_scope, out, encoder_output, encoder_output)

        out = tf.math.l2_normalize(out + multiHead_out, axis=-1)

        FF_out = self.build_FFNN(layer_scope, out)

        return out + FF_out

    def build_scaled_dotProduct_attention(self, Q, K, V, mask=False):

        Q_K = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (self.config.embedding_dim ** 0.5)

        if mask:
            diag_val = tf.ones_like(Q_K[0, :, :])

            tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()

            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(Q_K)[0], 1, 1])

            padding = tf.ones_like(masks) * (-math.pow(2, 32) + 1)

            Q_K = tf.where(tf.equal(masks, 0), padding, Q_K)

        out = tf.matmul(tf.nn.softmax(Q_K), V)

        return out

    def build_multiHead_attention(self, scope, Q, K, V, mask=False):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            heads = []

            for i in range(self.config.multi_head_h):
                dense_Q = tf.layers.Dense(self.config.k_dim, name="dense_Q_" + str(i))
                dense_K = tf.layers.Dense(self.config.k_dim, name="dense_K_" + str(i))
                dense_V = tf.layers.Dense(self.config.v_dim, name="dense_V_" + str(i))

                heads.append(self.build_scaled_dotProduct_attention(dense_Q(Q), dense_K(K), dense_V(V), mask=mask))

            multiHead = tf.concat(heads, axis=-1)

            dense_multiHead = tf.layers.Dense(self.config.embedding_dim, name="dense_multiHead")

            out = dense_multiHead(multiHead)

        return out

    def build_FFNN(self, scope, inputs):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            dense_FF_1 = tf.layers.Dense(self.config.embedding_dim * 2, name="dense_FF1")
            dense_FF_2 = tf.layers.Dense(self.config.embedding_dim, name="dense_FF2")

            out = dense_FF_2(tf.nn.relu(dense_FF_1(inputs)))

        return out

    def build(self):
        self.build_embedding()

        title2sent_dense = tf.layers.Dense(self.vocab_size, name="dense_title2sent")
        sent2sent_dense = tf.layers.Dense(self.vocab_size, name="dense_sent2sent")
        sent2next_dense = tf.layers.Dense(self.vocab_size, name="dense_sent2next")

        if self.mode == 'train':

            title2sent_encoder_output = self.build_encoder_layer('title2sent',
                                                                 self.title2sent_title,
                                                                 inputs_mask=self.title2sent_input_mask)
            title2sent_out = self.build_decoder_layer(title2sent_encoder_output,
                                                      'title2sent',
                                                      self.title2sent_sent)
            title2sent_out = title2sent_dense(title2sent_out)

            sent2sent_encoder_output = self.build_encoder_layer('sent2sent',
                                                                self.sent2sent_sent1)
            sent2sent_out = self.build_decoder_layer(sent2sent_encoder_output,
                                                     'sent2sent',
                                                     self.sent2sent_sent2)
            sent2sent_out = sent2sent_dense(sent2sent_out)

            sent2next_encoder_output = self.build_encoder_layer('sent2next',
                                                                self.sent2next_sent,
                                                                inputs_mask=self.sent2next_input_mask)
            sent2next_out = self.build_decoder_layer(sent2next_encoder_output,
                                                     'sent2next',
                                                     self.sent2next_sent3)
            sent2next_out = sent2next_dense(sent2next_out)

            title2sent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.title2sent_sent_target,
                                                                               logits=title2sent_out)
            sent2sent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sent2sent_sent2_target,
                                                                              logits=sent2sent_out)
            sent2next_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sent2next_sent3_target,
                                                                              logits=sent2next_out)
            self.cross_entropy_loss = [title2sent_losses, sent2sent_losses, sent2next_losses]

            title2sent_loss = tf.reduce_mean(title2sent_losses)
            sent2sent_loss = tf.reduce_mean(sent2sent_losses)
            sent2next_loss = tf.reduce_mean(sent2next_losses)

            self.total_loss = title2sent_loss + sent2sent_loss + sent2next_loss

            summary_loss = tf.summary.scalar("loss", self.total_loss)
            summary_title2sent_loss = tf.summary.scalar("title2sent_loss", title2sent_loss)
            summary_sent2sent_loss = tf.summary.scalar("sent2sent_loss", sent2sent_loss)
            summary_sent2next_loss = tf.summary.scalar("sent2next_loss", sent2next_loss)

            self.merge_summary = tf.summary.merge([summary_loss,
                                                   summary_title2sent_loss,
                                                   summary_sent2sent_loss,
                                                   summary_sent2next_loss])

            summary_eval_loss = tf.summary.scalar("eval_loss", self.eval_loss)
            summary_eval_perplexity = tf.summary.scalar("eval_perplexity", self.perplexity)

            self.eval_summary = tf.summary.merge([summary_eval_loss, summary_eval_perplexity])

        else:

            def decoding(encoder_input, scope, dense):

                decoder_sent = tf.constant([[0]])
                sent = []
                decode_sent_emb = tf.nn.embedding_lookup(self.embedding_map, decoder_sent)

                for i in range(5):
                    decoder_out = self.build_decoder_layer(encoder_input, scope, decode_sent_emb, mask_head=False)
                    decoder_out = tf.argmax(tf.nn.softmax(dense(decoder_out)), dimension=-1)
                    char_index = tf.gather_nd(decoder_out, [[0, i]])

                    sent.append(char_index)

                    decoder_sent = tf.concat([decoder_sent, [char_index]], axis=1)
                    decode_sent_emb = tf.nn.embedding_lookup(self.embedding_map, decoder_sent)

                return sent

            title_output = self.build_encoder_layer('title2sent', self.inference_input)
            sent1 = decoding(title_output, 'title2sent', title2sent_dense)
            sent1_emb = tf.nn.embedding_lookup(self.embedding_map, tf.expand_dims(tf.concat(sent1, axis=0), 0))

            sent1_output = self.build_encoder_layer('sent2sent', sent1_emb)
            sent2 = decoding(sent1_output, 'sent2sent', sent2sent_dense)
            sent2_emb = tf.nn.embedding_lookup(self.embedding_map, tf.expand_dims(tf.concat(sent2, axis=0), 0))

            sent_t_1_2 = tf.concat([self.inference_input, sent1_emb, sent2_emb], axis=1)
            sent3_output = self.build_encoder_layer('sent2next', sent_t_1_2)
            sent3 = decoding(sent3_output, 'sent2next', sent2next_dense)
            sent3_emb = tf.nn.embedding_lookup(self.embedding_map, tf.expand_dims(tf.concat(sent3, axis=0), 0))

            sent3_output = self.build_encoder_layer('sent2sent', sent3_emb)
            sent4 = decoding(sent3_output, 'sent2sent', sent2sent_dense)

            self.out = [sent1, sent2, sent3, sent4]



