import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten


# LOAD FEATURE VECTOR FROM .CSV FILES
path = "C:/Users/Paolo De Santis/Desktop/Repository - CMLS - HM1/Feature Vectors Archive/Test fold 9/"
x_train = np.genfromtxt(path + 'x_train.csv', delimiter=',')
x_test = np.genfromtxt(path + 'x_test.csv', delimiter=',')
y_train = np.genfromtxt(path + 'y_train.csv', delimiter=',')
y_test = np.genfromtxt(path + 'y_test.csv', delimiter=',')


# Encode the classification labels
le = LabelEncoder()
yy_train = to_categorical(le.fit_transform(y_train), num_classes=10)
yy_test = to_categorical(le.fit_transform(y_test), num_classes=10)

print(x_train.shape)
print(x_test.shape)
print(yy_train.shape)
print(yy_test.shape)

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
score = model.evaluate(x_test, yy_test, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 32


model.fit(x_train, yy_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, yy_test),  verbose=1)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, yy_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, yy_test, verbose=0)
print("Testing Accuracy: ", score[1])