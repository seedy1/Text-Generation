import numpy as np
# import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#read txt file

file_name = "alice.txt"
text_data = open(file_name, 'r', encoding='utf-8').read()
raw_text = text_data.lower()

# print(raw_text[:10])

#
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c,i) for i, c in enumerate(chars) )
n_chars = len(raw_text)
n_vocab = len(chars)

print('Total Cahracters: ', n_chars)
print('Total vocab: ', n_vocab)
#--------------------------------------
#input output for encoder as ints
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars-seq_length,1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([ chars_to_int[char] for char in seq_in ])
    dataY.append(chars_to_int[seq_out])
n_patterns = len(dataX)
print('Total Patterns: ', n_patterns)

# one hot encoding

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

print(X.shape)
print(y.shape)

# LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]) ))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print('------ HMMMMMMM.... -------')

# check points for initial loss
file_path = 'weights-improvement-{epoch:02f}-{loss:.4f}.hdf5'
checkpoints = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoints]

# fit fit fit
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callback_list)



