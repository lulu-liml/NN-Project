
import numpy as np
import pandas as pd
from google.colab import files
import io
import os
import json
import random
import nltk
import re
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score, mean_squared_error
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import LSTM

"""# Data Preparation
Upload the dataset from Kaggle 

https://www.kaggle.com/jkgatt/restaurant-data-with-100-trip-advisor-reviews-each?select=factual_tripadvisor_restaurant_data_all_100_reviews.json
"""

with open('C:/Users/limen/OneDrive/Desktop/factual_tripadvisor_restaurant_data_all_100_reviews.json',encoding='utf-8') as json_file:
    dataset = json.load(json_file)


reviews = []
ratings = []
restaurant_id = []
num_of_restaurant = 147

for i in range(num_of_restaurant):
    score = 0
    text = ""
    count = 0
    for review in dataset["restaurants"][i]["reviews"]:
        reviews.append(review["review_text"])
        ratings.append(review["review_rating"])
        restaurant_id.append(i)
        count = count + 1

data = pd.DataFrame(list(zip(ratings, reviews, restaurant_id)), columns =['Rating', 'Reviews', 'Restaurant_id']) 

train_data = pd.DataFrame(columns={"Rating","Reviews","Restaurant_id"}) 
test_data = pd.DataFrame(columns={"Rating","Reviews","Restaurant_id"}) 

selected_restaurant = random.sample(list(np.unique(data["Restaurant_id"])), int(num_of_restaurant*0.21))
for i in range(num_of_restaurant):
  if i not in selected_restaurant:
    train_data = train_data.append(data.loc[data['Restaurant_id'] == i])
  else:
    test_data = test_data.append(data.loc[data['Restaurant_id'] == i])



nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

def text_preprocess(review):
  review = re.sub(r'[^\x00-\x7F]+',' ', review)
  # no digit
  remove_digits = str.maketrans('', '', string.digits)
  review = review.translate(remove_digits)
  # no punctuation
  review = review.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
  # low case
  review = review.lower()
  # no multiple space
  review = re.sub(r'\s+', ' ', review)
  # no stop words
  bagOfWords = review.split()
  bagOfWords = [w for w in bagOfWords if not w in stop_words] 
  review = " ".join(bagOfWords)
  
  return review

#clean train set
train_x = []
train_y = list(train_data["Rating"])
train_id = list(train_data["Restaurant_id"])
train_text = list(train_data["Reviews"])
for i in train_text:
  train_x.append(text_preprocess(i))

#clean test set
test_x = []
test_y = list(test_data["Rating"])
test_id = list(test_data["Restaurant_id"])
test_text = list(test_data["Reviews"])
for i in test_text:
  test_x.append(text_preprocess(i))


tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_x)

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

max = 100
train_x = pad_sequences(train_x, padding = 'post', maxlen = max)
test_x = pad_sequences(test_x, padding = 'post', maxlen = max)

!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove.6B.zip

glove_dir = './'

embeddings_index = {} 

#here we choose the 100d pre-trained word vector
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 100
num_words = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim)) #create an array of zeros with word_num rows and embedding_dim columns
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < num_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

"""# Feed Forward Neural Network"""

#convert the label into one-hot coding
train_y = array(train_y)
encoded_train_y = to_categorical(train_y)

test_y = array(test_y)
encoded_test_y = to_categorical(test_y)


model = Sequential()
layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length = max, trainable = False)
model.add(layer)
model.add(Flatten())
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['acc'])
model.fit(train_x, encoded_train_y, batch_size = 200, epochs = 30, validation_split = 0.1)

print(model.summary())

"""Calculate the prediction performance"""

predict_restaurant_y = []
actual_restaurant_y = []

encoded_predict_y = model.predict(test_x)
predict_y = np.argmax(encoded_predict_y, axis=1)

sort = np.sort(selected_restaurant)
for i in range(len(sort)):
  p_y = sum(predict_y[i*100:(i+1)*100])
  p_y = p_y/100
  predict_restaurant_y.append(p_y)
  a_y = sum(test_y[i*100:(i+1)*100])
  a_y = a_y/100
  actual_restaurant_y.append(a_y)

print("R2: "+ str(r2_score(test_y, predict_y)))
print("mean_squared_error: "+ str(mean_squared_error(test_y, predict_y)))

"""# Convolutional Neural Network
"""

model = Sequential()
layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length = max, trainable = False)
model.add(layer)
model.add(Conv1D(embedding_dim, 5, activation = "relu"))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['acc'])
model.fit(train_x, encoded_train_y, batch_size = 200, epochs = 30, validation_split = 0.1)

print(model.summary())

"""Calculate the prediction performance"""

predict_restaurant_y = []
actual_restaurant_y = []

encoded_predict_y = model.predict(test_x)
predict_y = np.argmax(encoded_predict_y, axis=1)

sort = np.sort(selected_restaurant)
for i in range(len(sort)):
  p_y = sum(predict_y[i*100:(i+1)*100])
  p_y = p_y/100
  predict_restaurant_y.append(p_y)
  a_y = sum(test_y[i*100:(i+1)*100])
  a_y = a_y/100
  actual_restaurant_y.append(a_y)

print("R2: "+ str(r2_score(test_y, predict_y)))
print("mean_squared_error: "+ str(mean_squared_error(test_y, predict_y)))

"""# Recurrenct Neural Network (LSTM)
"""

model = Sequential()
layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length = max, trainable = False)
model.add(layer)
model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['acc'])
model.fit(train_x, encoded_train_y, batch_size = 200, epochs = 30, validation_split = 0.1)

print(model.summary())

"""Calculate the prediction performance"""

predict_restaurant_y = []
actual_restaurant_y = []

encoded_predict_y = model.predict(test_x)
predict_y = np.argmax(encoded_predict_y, axis=1)
sort = np.sort(selected_restaurant)
for i in range(len(sort)):
  p_y = sum(predict_y[i*100:(i+1)*100])
  p_y = p_y/100
  predict_restaurant_y.append(p_y)
  a_y = sum(test_y[i*100:(i+1)*100])
  a_y = a_y/100
  actual_restaurant_y.append(a_y)

print("R2: "+ str(r2_score(test_y, predict_y)))
print("mean_squared_error: "+ str(mean_squared_error(test_y, predict_y)))

"""# Reference
[1]J. Pennington, GloVe: Global Vectors for Word Representation. [Online]. Available: https://nlp.stanford.edu/projects/glove/.
"""