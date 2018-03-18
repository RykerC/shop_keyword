from gensim.models import Word2Vec
import pre_process as pp
import pandas as pd
import jieba
import re

segment_sentence_train, length_train, label_vector, segment_sentence_test, length_test, length_label, l1,l2,l3 = pp.clean_files('train.csv', 'test.csv')

df = pd.read_csv('train.csv',
                     names=['product name', 'category', 'query', 'event', 'date'])
df2 = pd.read_csv('test.csv',
                  names=['product name', 'category'])

query_list = df['query'].tolist()
labels = []

for i in query_list:
    labels.append([i])
query_word_list = [sentence.split(' ') for sentence in query_list]

for ls in query_word_list:
    for word in ls:
        jieba.add_word(word)
product_name_train = df['product name'].tolist()
product_name_test = df2['product name'].tolist()

segment_sentence = []
temp_list1 = [''.join(re.split(r'\W+', sentence.lower())) for sentence in product_name_train]
temp_list2 = [''.join(re.split(r'\W+', sentence.lower())) for sentence in product_name_test]

lens = []
after_cut_list_train = []
after_cut_list_test = []

for sentence in temp_list1:
    text = jieba.lcut_for_search(sentence)
    text_length = len(text)
    lens.append(text_length)
    after_cut_list_train.append(text)
for sentence in temp_list2:
    text = jieba.lcut_for_search(sentence)
    text_length = len(text)
    lens.append(text_length)
    after_cut_list_test.append(text)
model = Word2Vec(labels+after_cut_list_train+after_cut_list_test+l1+l2+l3+query_word_list, min_count=1,size=400,workers=4)

model.save('word_embedding.wv')

