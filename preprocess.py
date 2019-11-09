from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from langconv import *
import json
import random


# def check_vocab(sent, vocab):
#
#     sent_arr = []
#     for c in sent:
#         if not vocab.HasWord(c):
#             vocab.word2id[c] = vocab.nextID
#             vocab.id2word[vocab.nextID] = c
#             vocab.nextID += 1
#         sent_arr.append(vocab.GetID(c))
#     return sent_arr


def build_data_files():

    file_names = ["poet.tang.0.json"]

    for i in range(1, 58):
        file = "poet.tang." + str(i) + "000.json"
        file_names.append(file)

    return file_names


def build_input_data(data_files):
    """
    Read data from files
    Split data into training and validation sets

    T2S_pair = [ [title, sent],
                 [title, sent], ...]

    Sent1ToSent2 = [ [title, sent1, sent2],
                     [title, sent1, sent2], ...]

    Sent12ToSent2 = [ [title, sent1, sent2, sent3],
                      [title, sent1, sent2, sent3], ...]
    """
    title2sent = []
    sent2sent = []
    sent2next = []

    vocab = ['<start>', '<end>', '<pad>']
    spe_c = ' 1234567890`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

    for file in data_files:
        file = 'data/' + file
        with open(file, 'r', encoding='utf-8') as f:
            d = json.load(f)

            for s in d:
                if len(s['paragraphs']) == 0 or len(s['title']) == 0:
                    continue
                num_sents = len(s['paragraphs'])
                valid = True
                for i in range(num_sents):
                    if (len(s['paragraphs'][i])//2 - 1) != 5:
                        valid = False
                        break
                if not valid:
                    continue
                title = Converter('zh-hans').convert(s['title'])
                title = title.split()[0]
                title_s = []
                for c in title:
                    if c not in vocab:
                        vocab.append(c)
                    title_s.append(c)
                # title = check_vocab(title, vocab)

                sent = Converter('zh-hans').convert(s['paragraphs'][0][:5])
                # sent = check_vocab(sent, vocab)
                sent_s = []
                for c in sent:
                    if c not in vocab:
                        vocab.append(c)
                    sent_s.append(c)
                sent_start = ['<start>'] + sent_s[:]
                sent_end = sent_s[:] + ['<end>']

                title2sent.append([title_s, sent_start, sent_end])
                for i in range(num_sents-1):
                    sent1 = Converter('zh-hans').convert(s['paragraphs'][i][:5])
                    # sent1 = check_vocab(sent1, vocab)
                    sent1_s = []
                    for c in sent1:
                        if c not in vocab:
                            vocab.append(c)
                        sent1_s.append(c)
                    if len(sent1_s) != 5:
                        print("ssssssssssssssssssssssssssssssssssssss")
                        continue
                    sent2 = Converter('zh-hans').convert(s['paragraphs'][i][5+1:-1])
                    # sent2 = check_vocab(sent2, vocab)
                    sent2_s = []
                    for c in sent2:
                        if c not in vocab:
                            vocab.append(c)
                        sent2_s.append(c)
                    if len(sent2_s) != 5:
                        print("ssssssssssssssssssssssssssssssssssssss")
                        continue
                    sent3 = Converter('zh-hans').convert(s['paragraphs'][i+1][:5])
                    # sent3 = check_vocab(sent3, vocab)
                    sent3_s = []
                    for c in sent3:
                        if c not in vocab:
                            vocab.append(c)
                        sent3_s.append(c)
                    if len(sent3_s) != 5:
                        print("ssssssssssssssssssssssssssssssssssssss")
                        continue

                    sent2_start = ['<start>'] + sent2_s[:]
                    sent2_end = sent2_s[:] + ['<end>']

                    sent3_start = ['<start>'] + sent3_s[:]
                    sent3_end = sent3_s[:] + ['<end>']

                    sent2sent.append([title_s, sent1_s, sent2_start, sent2_end])
                    sent2next.append([title_s, sent1_s, sent2_s, sent3_start, sent3_end])

                sent1 = Converter('zh-hans').convert(s['paragraphs'][num_sents-1][:5])
                # sent1 = check_vocab(sent1, vocab)
                sent1_s = []
                for c in sent1:
                    if c not in vocab:
                        vocab.append(c)
                    sent1_s.append(c)
                sent2 = Converter('zh-hans').convert(s['paragraphs'][num_sents-1][5+1:-1])
                # sent2 = check_vocab(sent2, vocab)
                sent2_s = []
                for c in sent2:
                    if c not in vocab:
                        vocab.append(c)
                    sent2_s.append(c)

                sent2_start = ['<start>'] + sent2_s[:]
                sent2_end = sent2_s[:] + ['<end>']

                sent2sent.append([title_s, sent1_s, sent2_start, sent2_end])

    with open('model/vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')

    return title2sent, sent2sent, sent2next, len(vocab)


def build_embedding_map(embedding_file, vocab_file):
    """
    Create the embedding map

    Return:
        Embedding map for words in the vocabulary file
        - Random Normal for unknown words
        - [0...] for <pad>
    """

    words_dic = {}

    with open(embedding_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split()
            embedding = []
            for ele in elements[1:]:
                embedding.append(float(ele))
            words_dic[elements[0]] = embedding

    embedding_map = []
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        for word in lines:
            if word[:1] in words_dic:
                embedding_map.append(words_dic[word[:-1]])
            else:
                embedding_map.append([random.uniform(-1.0, 1.0) for _ in range(300)])
        embedding_map.append([random.uniform(-1.0, 1.0) for _ in range(300)])
    return embedding_map


def build_input_batch(title2sent, sent2sent, sent2next, batch_size, model):

    title2sent_title = []
    title2sent_sent = []
    title2sent_sent_target = []
    title2sent_title_length = []

    sent2sent_sent1 = []
    sent2sent_sent2 = []
    sent2sent_title = []
    sent2sent_sent2_target = []
    sent2sent_title_length = []

    sent2next_sent1 = []
    sent2next_sent2 = []
    sent2next_sent3 = []
    sent2next_title = []
    sent2next_sent3_target = []
    sent2next_title_length = []

    for i in range(batch_size):
        title, start_sent, end_sent = title2sent.pop(0)
        title2sent.append([title, start_sent, end_sent])

        title2sent_title.append(title[:])
        title2sent_sent.append(start_sent[:])
        title2sent_sent_target.append(end_sent[:])
        title2sent_title_length.append(len(title))

        title1, sent1, start_sent2, end_sent2 = sent2sent.pop(0)
        sent2sent.append([title1, sent1, start_sent2, end_sent2])
        sent2sent_sent1.append(sent1[:])
        sent2sent_sent2.append(start_sent2[:])
        sent2sent_title.append(title1[:])
        sent2sent_sent2_target.append(end_sent2[:])
        sent2sent_title_length.append(len(title1))

        title2, sent1, sent2, start_sent3, end_sent3 = sent2next.pop(0)
        sent2next.append([title2, sent1, sent2, start_sent3, end_sent3])
        sent2next_sent1.append(sent1[:])
        sent2next_sent2.append(sent2[:])
        sent2next_sent3.append(start_sent3[:])
        sent2next_title.append(title2[:])
        sent2next_sent3_target.append(end_sent3[:])
        sent2next_title_length.append(len(title2))

    max1 = max(title2sent_title_length)
    for i in range(batch_size):
        for _ in range(max1-title2sent_title_length[i]):
            title2sent_title[i].append('<pad>')

    max2 = max(sent2sent_title_length)
    for i in range(batch_size):
        for _ in range(max2 - sent2sent_title_length[i]):
            sent2sent_title[i].append('<pad>')

    max3 = max(sent2next_title_length)
    for i in range(batch_size):
        for _ in range(max3-sent2next_title_length[i]):
            sent2next_title[i].append('<pad>')

    return {model.title2sent_title_holder: title2sent_title,
            model.title2sent_sent_holder: title2sent_sent,
            model.title2sent_sent_target_holder: title2sent_sent_target,
            model.title2sent_title_length: title2sent_title_length,
            model.sent2sent_sent1_holder: sent2sent_sent1,
            model.sent2sent_sent2_holder: sent2sent_sent2,
            model.sent2sent_title_holder: sent2sent_title,
            model.sent2sent_sent2_target_holder: sent2sent_sent2_target,
            model.sent2sent_title_length: sent2sent_title_length,
            model.sent2next_sent1_holder: sent2next_sent1,
            model.sent2next_sent2_holder: sent2next_sent2,
            model.sent2next_sent3_holder: sent2next_sent3,
            model.sent2next_title_holder: sent2next_title,
            model.sent2next_sent3_target_holder: sent2next_sent3_target,
            model.sent2next_title_length: sent2next_title_length
            }, title2sent, sent2sent, sent2next
