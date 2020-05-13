import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
# from playsound import playsound

# DATA COLLECTION

# path of UrbanSound8K 10 folds
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/"

# Read metadata file
data = pd.read_csv(path + "UrbanSound8K/metadata/UrbanSound8K.csv")

# Reorganize data for fold n 10
data_1fold = []

for i in tqdm(range(len(data))):
    fold_no = str(data.iloc[i]["fold"])
    if fold_no == '10':
        data_1fold.append(data.iloc[i])

d1f = pd.DataFrame(data_1fold)

# FEATURE EXTRACTION

# Number of file for each class in fold
print(d1f["classID"].value_counts())
n_samples_for_classes = d1f["classID"].value_counts()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dict_train_features = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
n_coeff = 40

for c in classes:
    n_train_samples = n_samples_for_classes[int(c)]
    train_features = np.zeros((n_train_samples, n_coeff))
    j = 0

    for i in tqdm(range(len(d1f))):
        label = d1f.iloc[i]["classID"]

        if str(label) == c:
            fold_no = str(d1f.iloc[i]["fold"])
            file = d1f.iloc[i]["slice_file_name"]
            filename = path + "UrbanSound8K/audio/fold" + fold_no + "/" + file
            y, sr = librosa.load(filename, mono=True)  # convert to mono

            # if c == '4' and (j == 42 or j == 43 or j == 44 or j == 45)):
            #     playsound(filename)

            # MEL Feature
            mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=n_coeff).T, axis=0)
            # melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_coeff, fmax=8000).T, axis=0)

            # Chroma Feature
            # chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_coeff).T, axis=0)
            # chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_coeff).T, axis=0)
            # chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=n_coeff).T, axis=0)

            train_features[j, :] = mfccs
            j += 1

    dict_train_features[c] = train_features

# VISUALIZATION

for c in classes:
    feature = dict_train_features[c].transpose()
    fig = plt.figure(figsize=(16, 6))
    plt.imshow(feature, origin='lower', aspect='auto')
    plt.xlabel('Training samples')
    plt.ylabel('MFCC coefficients')
    plt.title('MFCC (40 coefficients) for class {}'.format(c))
    plt.colorbar()
    plt.tight_layout()

plt.show()

