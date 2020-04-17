import pandas as pd
import numpy as np
import librosa
import sklearn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# DATA HANDLING
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/"

# Read metadata file
data = pd.read_csv(path + "UrbanSound8K/metadata/UrbanSound8K.csv")

#Reorganize data for the first 3 folds
data_3folds = []
for i in tqdm(range(len(data))):
    fold_no = str(data.iloc[i]["fold"])
    if (fold_no == '1' or fold_no == '2' or fold_no == '3'):
        data_3folds.append(data.iloc[i])
        
d3f = pd.DataFrame(data_3folds)

# FEATURE EXTRACTION
x_train = []  # feature coeffs
x_test = []  # feature coeffs
y_train = []  # labels
y_test = []  # labels (desidered outputs)

test_fold = '3'

for i in tqdm(range(len(d3f))):
    fold_no = str(d3f.iloc[i]["fold"])
    file = d3f.iloc[i]["slice_file_name"]
    label = d3f.iloc[i]["classID"]
    filename = path + "UrbanSound8K/audio/fold" + fold_no + "/" + file
    y, sr = librosa.load(filename, mono=True)  # convert to mono

    # MEL Feature
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
    #melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    #
    # #Chroma Feature
    #chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    #chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
    #chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
    #
    #features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5)) #!!! 40 is the number of coeffs
    #features = np.reshape(np.vstack((mfccs, melspectrogram)), (40, 2))

    if (fold_no != test_fold):
        x_train.append(mfccs)  # features
        y_train.append(label)
    else:
        x_test.append(mfccs)  # features
        y_test.append(label)

# Converting the lists into numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# nsamples_train, nx_train, ny_train = x_train.shape
# nsamples_test, nx_test, ny_test = x_test.shape
# d2_x_train = x_train.reshape((nsamples_train,nx_train*ny_train))
# d2_x_test = x_test.reshape((nsamples_test,nx_test*ny_test))

# # Normalize features
# feat_max = np.max(x_train, axis=0) #x_train/d2_x_train
# feat_min = np.min(x_train, axis=0) #x_train/d2_x_train
# x_train_normalized = (x_train - feat_min) / (feat_max - feat_min) #x_train/d2_x_train
# x_test_normalized = (x_test - feat_min) / (feat_max - feat_min) #x_test/d2_x_test

# Support Vector Machine Classifier
clf = sklearn.svm.SVC(C=1, kernel='rbf')
clf.fit(x_train, y_train) #x_train_normalized/x_train
y_predict = clf.predict(x_test)#x_test_normalized/x_test

#Evaluation
accuracy = clf.score(x_test, y_test)#x_test_normalized/x_test
print("Accuracy:")
print(accuracy)
print(confusion_matrix(y_test, y_predict))

