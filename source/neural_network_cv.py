import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split


# LOAD FEATURE VECTOR FROM .CSV FILES
path = "C:/Users/Paolo De Santis/Desktop/Repository - CMLS - HM1/Feature Vectors Archive/Test fold 9/"

xx_train = pd.read_csv(path + 'x_train.csv', sep=',', dtype=float)
xx_test = pd.read_csv(path + 'x_test.csv', sep=',', dtype=float)
yy_train = pd.read_csv(path + 'y_train.csv', sep=',', dtype=int)
yy_test = pd.read_csv(path + 'y_test.csv', sep=',', dtype=int)

# Dataframe to numpy array
x_train = xx_train.to_numpy()
x_test = xx_test.to_numpy()
yyy_train = yy_train.to_numpy()
yyy_test = yy_test.to_numpy()

# Encode the classification labels
y_train = to_categorical(yyy_train, num_classes=10)
y_test = to_categorical(yyy_test, num_classes=10)

num_labels = 10
filter_size = 2

model = Sequential()

model.add(Dense(256, input_shape=(40,)))
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