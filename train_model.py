import numpy as np
import re
import string
#
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
#
import spacy

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, to_categorical

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pickle import dump, load


#
# from random import randint
# from bs4 import BeautifulSoup
# from keras.models import load_model

# load data
def read_file(file_name):
    with open(file_name) as f:
        text = f.read()
    return text


nlp.max_length = 1198623
# print(read_file('moby_dick_four_chapters.txt'))
file_name = "moby_dick_four_chapters.txt"
text_data = open(file_name, 'r', encoding='utf-8').read()
# raw_text = text_data.lower()
# print(text_data)
print(string.punctuation)


# clean the text
def clean_text1(text):
    text = text.lower()
    text = re.sub('\[.*\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # removes punctuation and other symbol
    text = re.sub('\n\n\n', '', text)
    text = re.sub('\n\n', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)  # remove numbers
    # text = re.sub('  ', '', text)

    return text


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if
            token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


d = read_file('moby_dick_four_chapters.txt')
tokens = separate_punc(d)

# print(tokens)
#
# def get_words(text):
#     for word in clean_text:
#         if word.isalpha():
#             return word


clean_text = clean_text1(text_data)
# clean_text = get_words(clean_text)
# print(len(clean_text))

# creating a sequence of the tokens

train_length = 30 + 1
text_seq = []

for i in range(train_length, len(tokens)):
    seq = tokens[i - train_length:i]
    text_seq.append(seq)

print(' '.join(text_seq[0]))
# print(' '.join(text_seq[1]))
# print(' '.join(text_seq[2]))

print(len(text_seq))

# tokenizie the clen text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_seq)
sequences = tokenizer.texts_to_sequences(text_seq)

# print(sequences[0])

for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')

vocabulary_size = len(tokenizer.word_counts)
sequences = np.array(sequences)

# create sequeuces with final word 'y'
X = sequences[:, :-1]  # first n-1 words
# print(sequences.shape)
# print('sssssaaaa  :', sequences[0:1])


# print('sssss  :',sequences[0,-1])
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocabulary_size + 1)
print(X.shape[1])
seq_length = X.shape[1]


# ------- O R -------------#
# #
# chars = sorted(list(set(clean_text)))
# chars_to_int = dict((c, i) for i, c in enumerate(chars))
# n_chars = len(clean_text)
# n_vocab = len(chars)
#
# print('Total Cahracters: ', n_chars)
# print('Total vocab: ', n_vocab)
#
# seq_length = 100
# dataX = []
# dataY = []
# for i in range(0, n_chars - seq_length, 1):
#     seq_in = clean_text[i:i + seq_length]
#     seq_out = clean_text[i + seq_length]
# print('IN: ', seq_in)
# print('OUT: ', seq_out)
#     dataX.append([chars_to_int[char] for char in seq_in])
#     dataY.append(chars_to_int[seq_out])
# n_patterns = len(dataX)
# print('Total Patterns: ', n_patterns)
#
# # one hot encoding
#
# X = np.reshape(dataX, (n_patterns, seq_length, 1))
# X = X / float(n_vocab)
# y = np_utils.to_categorical(dataY)
#
# print(X.shape)
# print(y.shape)

# vocab_size=128 + 1
# ------------#

def init_model(vocabulary_size, sequeunce_length):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 25, input_length=sequeunce_length))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


# SET UP AND TRAINING
model = init_model(vocabulary_size + 1, seq_length)
#
# model.fit(X, y, batch_size=128, epochs=1, verbose=1)

#save model after running
# model.save('myModel.h5')
# dump(tokenizer, open('myModel', 'wb'))

print('THAT\'S ALL FLOCKS')