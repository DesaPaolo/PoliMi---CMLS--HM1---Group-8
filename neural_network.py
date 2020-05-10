import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten


# LOAD FEATURE VECTOR FROM .CSV FILES
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
x_train = np.genfromtxt(path + '1-6,8-10folds_mfcc_n_coeff=13_x_train.csv', delimiter=',')
x_test = np.genfromtxt(path + 'fold7_mfcc_n_coeff=13_x_test.csv', delimiter=',')
y_train = np.genfromtxt(path + '1-6,8-10folds_y_train.csv', delimiter=',')
y_test = np.genfromtxt(path + 'fold7_y_test.csv', delimiter=',')

num_labels = 10

model = Sequential()

model.add(Dense(256, input_shape=(13,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 32


model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),  verbose=1)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])