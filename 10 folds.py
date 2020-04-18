import pandas as pd
import numpy as np
import librosa
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# DATA HANDLING
path = "/Users/PilvioSol/Desktop/UrbanSound8K"

# Read metadata file
data = pd.read_csv(path + "/metadata/UrbanSound8K.csv")

# Number of file in each fold
# print(data["fold"].value_counts())

# FEATURE EXTRACTION
x_train = []  # feature coeffs
x_test = []  # feature coeffs
y_train = []  # labels
y_test = []  # labels (desidered outputs)

test_fold = '1'

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]]

for i in tqdm(range(len(data))):
    fold_no = str(data.iloc[i]["fold"])
    file = data.iloc[i]["slice_file_name"]
    label = data.iloc[i]["classID"]
    filename = path + "/audio/fold" + fold_no + "/" + file
    y, sr = librosa.load(filename)  # convert to mono
    y= parse_audio(y)
    # MEL Feature
    mfccs = np.mean(librosa.feature.mfcc(y, sr=sr).T, axis=0)
    # melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    #
    # #Chroma Feature
    # chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    # chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
    # chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
    #
    # features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5)) #!!! 40 is the number of coeffs

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

# Normalize features
# feat_max = np.max(x_train, axis=0)
# feat_min = np.min(x_train, axis=0)
# x_train_normalized = (x_train - feat_min) / (feat_max - feat_min)
# x_test_normalized = (x_test - feat_min) / (feat_max - feat_min)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Support Vector Machine Classifier
# clf = sklearn.svm.SVC(C=1, kernel='rbf')
# clf.fit(x_train_normalized, y_train)
# y_predict = clf.predict(x_test_normalized)

grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)


#clf = sklearn.svm.SVC(C=1, kernel='rbf')
#clf.fit(x_train, y_train) #x_train_normalized/x_train
y_predict = model.predict(x_test_scaled)#x_test_normalized/x_test



#Evaluation
accuracy = model.score(x_test_scaled, y_test)#x_test_normalized/x_test
print("Accuracy:")
print(accuracy)
print(confusion_matrix(y_test, y_predict))


# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music
