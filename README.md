# shopee_keyword_extraction

## Overall Method

It is a keyword extraction task for searching on the website. It is a NLP task. The dataset we have is in Traditional Chinese. Chinese is quite different from English.

When we process Chinese, we need to do the segment first. After this, we need to convert the words into vectors. Then we build a proper neural network to do the training and prediction.

## Step by Step

### 1. Analyze the dataset

The first thing we need to do is analyze the raw data. The train set has these columns:
'product name', 'category', 'query', 'event', 'date'.
And the test set has:
'product name', 'category'.
Actually, based on the 'category', we could build several models, which means for each category it has its own model. But due to the limited time, I just build a general model for all categories.

The 'product name' is the input data that we need to use. and the 'query' can be treated as label. The 'event' indicates the customers behaviors toward the search results. 'date' is the timestamp.

### 2. Segment

Chinese is not like English, in which the words in sentence have a space to split. So when we deal with the Chinese, we need to do the Segment first. Segment can convert the whole sentence into several pharse or words. One of the most popular and useful tools that are used in Chinese Segment is called 'jieba'. In this project, I used this as my segment lib.

However, the traditional 'jieba' lib preforms well on Simplified Chinese, the date set we have is in Traditional Chinese. Under this situation, I found a dict that are better for Traditional Chinese. In the folder called 'dict.txt.big.txt'.
