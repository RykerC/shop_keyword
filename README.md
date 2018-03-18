# shopee_keyword_extraction

## Overall Method

It is a keyword extraction task for searching on the website. It is a NLP task. The dataset we have is in Traditional Chinese. Chinese is quite different from English.

When we process Chinese, we need to do the segment first. After this, we need to convert the words into vectors. Then we build a proper neural network to do the training and prediction.

## Step by Step

### 1. Analyze the dataset

The first thing we need to do is analyze the raw data. The train set has these columns:

`product name`, `category`, `query`, `event`, `date`.

And the test set has:

`product name`, `category`.

Actually, based on the `category`, we could build several models, which means for each category it has its own model. But due to the limited time, I just build a general model for all categories.

The `product name` is the input data that we need to use. and the `query` can be treated as label. The `event` indicates the customers behaviors toward the search results. `date` is the timestamp.

### 2. Clean the Data

In this part, we need to remove those meaningless symbols.

### 3. Segment

Chinese is not like English, in which the words in sentence have a space to split. So when we deal with the Chinese, we need to do the Segment first. Segment can convert the whole sentence into several pharse or words. One of the most popular and useful tools that are used in Chinese Segment is called `jieba`. In this project, I used this as my segment lib.

However, the traditional `jieba` lib preforms well on Simplified Chinese, the date set we have is in Traditional Chinese. Under this situation, I found a dict that are better for Traditional Chinese. In the folder called `dict.txt.big.txt`. So at the beginning of the code, we need to use `jieba.set_dictionary()` to change the segment dictionary.

Beside, the keyword in `query` may be not an option in the dictionary, so we add the `query` words list to our dictionary by using `jieba.add_word()`. So that when we do the segment, we can get more accurate results.

### 4. Word Embedding

The text data cannot be used directly in the neural network. We need to convert it into vectors first. And there are several ways to achieve this. One is `TfidfVectorizer` which is counting the frequency of the word tokens. And another way is `Word Embedding`. 

And in the `Word Embedding`, we have two main methods, one is `one-hot model`, the other is `Word2Vec`. I adopted the second one because it could consider the relationship(the distance) between words which in `one-hot model` is not considered. Also the `Word2Vec` can reduce the dimension of the word vector.

I used `gensim` module do the word embedding. When do the embedding, it needs corpus. I did't find much Traditional Chinese corpus I only used the train & test set to train the word embedding model, so the word embedding results are not that perfect.

After embedding, we could get a word2vec model, which I saved as `word_embedding.wv`. Then we could use `model.wv[word]` to get the vector representation of the word after we load the word2vec model by `model = Word2Vec.load('word_embedding.wv')`.

### 5. Sequence to Sequence Model

After the word embedding, we can start to build the neural network to do the trainin to get the keyword extraction. Based on the data, I treat this as a supervised learning task. In this part, I checked several method, but I think the Sequence to Sequence Model could have a good performance. The traditional classification usually used when we have fixed categories. But in the project, the keywords are always changed based on different product name. And in the model, the basic NN I used is LSTM(GRU gate actually).

LSTM is good at dealing with the text data since it has 'memory'. And GRU is better because it could be easier to get convergent.


### 6. Reverse the Vector to Words

After traing and prediction, we could get the results but in vectors. Now we use the 

`wv_model = Word2Vec.load('word_embedding.wv')

wv_model.most_similar(positive=[vector], topn=1)`

to get the most similar word towords the vector we got. 


## Execute

The code is based on Python3.

1. Run the `create_embedding.py` file to get the word2vec model `word_embedding.wv`.

2. Run the `neural_network.py` to get the output, the `result.csv` file





