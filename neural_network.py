import pre_process as pp
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from gensim.models import Word2Vec
import pandas as pd



train_X, input_length, label, test, output_length, label_length = pp.clean_files('train.csv', 'test.csv')


def build_model(input_shape, max_out_seq_len, hidden_size):

    model = Sequential()
    model.add(GRU(hidden_size[0], input_shape=(input_shape[1], input_shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_size[1], activation="relu"))
    model.add(RepeatVector(max_out_seq_len))
    model.add(GRU(hidden_size[1], return_sequences=True))
    model.add(TimeDistributed(Dense(units=input_shape[2], activation="linear")))
    model.compile(loss="mse", optimizer='adam')
    model.summary()

    return model



input_shape = train_X.shape
y_shape = label.shape


neurons = [128, 64]
model = build_model(input_shape, label_length, neurons)
model.fit(train_X, label, batch_size=300, epochs=150, verbose=1)
pred = model.predict(test)

print(pred)
print(pred.shape)
wv_model = Word2Vec.load('word_embedding.wv')
result = []
for words in pred:
    tem_ls = []
    for i in words:
        new_words = []
        word = wv_model.most_similar(positive=[i], topn=1)
        for j in word:
            if j[-1] > 0.95:
                new_words.append(j)
        if new_words != []:
            tem_ls.append(new_words)
    result.append(tem_ls)
df = pd.read_csv('test.csv', names=['product name', 'category'])
df['key word'] = result
df.to_csv('result.csv', encoding='utf-8')
print(result)
