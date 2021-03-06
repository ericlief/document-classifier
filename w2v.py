from __future__ import print_function
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD

word_index = reuters.get_word_index(path="reuters_word_index.json")
idx_to_word = dict(zip(word_index.values(), word_index.keys()))

max_words = 20706
batch_size = 32
epochs = 5

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
(x_train, y_train), (x_test, y_test) = reuters.load_data(test_split=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#print(x_train[1], y_train[1])
#print("words")
#words = [idx_to_word[i] for i in x_train[1]]
#print(" ".join(words))


num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')

#print(x_train[0], y_train[0])


x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


print('Building model...')
model = Sequential()
model.add(Dense(int(.5*max_words), input_shape=(max_words,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

"""
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(int(.25*max_words), activation='relu', input_dim=max_words))
model.add(Dropout(0.5))
model.add(Dense(int(.25*max_words), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])

model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
"""



with open('results-keras-1h-sig' + str(max_words) + '.txt', 'wt') as f:

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    f.write("Test score " + str(score[0]) + '\n')
    f.write("Test acc " + str(score[1]) + '\n')
