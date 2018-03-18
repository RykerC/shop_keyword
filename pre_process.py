import pandas as pd
import jieba
import re
import numpy as np
from gensim.models import Word2Vec


jieba.set_dictionary('dict.txt.big.txt')
model = Word2Vec.load('word_embedding.wv')


def embedding_map(product_name):
    # product_name = df['product name'].tolist()
    segment_sentence = []
    temp_list = [''.join(re.split(r'\W+', sentence.lower())) for sentence in product_name]
    lens = []
    extra_list = []
    after_cut_list = []
    for sentence in temp_list:
        text = jieba.lcut_for_search(sentence)
        text_length = len(text)
        lens.append(text_length)
        after_cut_list.append(text)
    max_length = max(lens)
    for sent_list in after_cut_list:
        tem_word_list = []
        for word in sent_list:
            try:
                word_vector = model.wv[word]
                tem_word_list.append(word_vector)
                # print(word_vector)
            except:
                print(word)
                extra_list.append([word])
        sent_length = len(tem_word_list)
        if sent_length < max_length:
            for i in range(max_length - sent_length):
                tem_word_list.append(np.zeros(400))

        segment_sentence.append(np.array(tem_word_list))
    return np.array(segment_sentence), max_length, extra_list

def clean_files(train, test):
    df = pd.read_csv(train,
                     names=['product name', 'category', 'query', 'event', 'date'])

    product_name_train = df['product name'].tolist()

    query_list = df['query'].tolist()

    query_word_list = [sentence.split(' ') for sentence in query_list]
    for ls in query_word_list:
        for word in ls:
            jieba.add_word(word)

    segment_sentence_train, length_train, extra_list1 = embedding_map(product_name_train)

    label_vector, length_label, extra_list3 = embedding_map(query_list)
    # for i in query_list:
    #     label_vector.append(model.wv[i])



    # category_list = df['category'].tolist()
    # category_label = []
    # for i in category_list:
    #     if i == 'Male Fashion':
    #         category_label.append(1)
    #     elif i == 'Female Fastion':
    #         category_label.append(2)
    #     elif i == 'Mobile & Gadgets':
    #         category_label.append(3)
    #
    # event_list = df['event'].tolist()
    # event_label = []
    # for i in event_list:
    #     if i == 'Impression':
    #         event_label.append(1)
    #     else:
    #         event_label.append(0)


    df2 = pd.read_csv(test)

    product_name_test = df2['Product Name'].tolist()

    segment_sentence_test, length_test, extra_list2 = embedding_map(product_name_test)


    return segment_sentence_train, length_train, label_vector, segment_sentence_test, length_test, length_label , extra_list1, extra_list2, extra_list3

