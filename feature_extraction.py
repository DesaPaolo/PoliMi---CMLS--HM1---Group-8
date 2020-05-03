import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyAudioAnalysis import ShortTermFeatures
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

# DATA HANDLING
path = "/Users/PilvioSol/Desktop/UrbanSound8K" #"C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K"

# Read metadata file
df = pd.read_csv(path + "/metadata/UrbanSound8K.csv")

# Reorganize df (choose folds in if) ---> if you want 10 folds comment this for
data = []
for i in tqdm(range(len(df))):
    fold_no = str(df.iloc[i]["fold"])
    if (fold_no == '1' or fold_no == '2' or fold_no == '3' or fold_no == '4' or fold_no == '5' or fold_no == '6' or fold_no == '7' or fold_no == '8' or fold_no == '9' or fold_no == '10'):
        data.append(df.iloc[i])

df = pd.DataFrame(data)

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# FEATURE EXTRACTION
x_train0 = []  # feature coeffs
x_test0 = []  # feature coeffs
y_train0 = []  # labels
y_test0 = []  # labels (desidered outputs)

x_train1 = []  # feature coeffs
x_test1 = []  # feature coeffs
y_train1 = []  # labels
y_test1 = []  # labels (desidered outputs)

x_train2 = []  # feature coeffs
x_test2 = []  # feature coeffs
y_train2 = []  # labels
y_test2 = []  # labels (desidered outputs)

x_train3 = []  # feature coeffs
x_test3 = []  # feature coeffs
y_train3 = []  # labels
y_test3 = []  # labels (desidered outputs)

x_train4 = []  # feature coeffs
x_test4 = []  # feature coeffs
y_train4 = []  # labels
y_test4 = []  # labels (desidered outputs)

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]]

test_fold = '7'

for i in tqdm(range(len(df))):
    fold_no = str(df.iloc[i]["fold"])
    file = df.iloc[i]["slice_file_name"]
    label = df.iloc[i]["classID"]
    filename = path + "/audio/fold" + fold_no + "/" + file
    y, sr = librosa.load(filename)  # convert to mono
    y = parse_audio(y)
    y1 = butter_lowpass_filter(y, 2000, 44100, 6)
    y2 = butter_lowpass_filter(y, 1500, 44100, 6)
    y3 = butter_lowpass_filter(y, 1000, 44100, 6)
    y4 = butter_lowpass_filter(y, 600, 44100, 6)

    # MFFCs
    #mfccs = np.mean(librosa.feature.mfcc(y, sr).T, axis=0)  # n_mfcc = 20 (default value)
    #chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=20).T, axis=0)
    #spectr_bandwidth =np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    mfccs0 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=13).T, axis=0)
    mfccs1 = np.mean(librosa.feature.mfcc(y1, sr, n_mfcc=13).T, axis=0)
    mfccs2 = np.mean(librosa.feature.mfcc(y2, sr, n_mfcc=13).T, axis=0)
    mfccs3 = np.mean(librosa.feature.mfcc(y3, sr, n_mfcc=13).T, axis=0)
    mfccs4 = np.mean(librosa.feature.mfcc(y4, sr, n_mfcc=13).T, axis=0)
    # sp_centr = librosa.feature.spectral_centroid(y,sr)[0]
    # sp_centr= sp_centr[:1]
    # sp_centr= [sp_centr,0,0,0,0,0,0,0,0,0,0,0,0]
    #sp_centr= np.repeat(sp_centr,13)
    #melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    #chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    #chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
    #chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
    #features=[mfccs,melspectrogram]
    #melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20,fmax=8000).T,axis=0)
    #features = np.reshape(np.vstack((mfccs,sp_centr)), (13, 2))
    # F, f_names = ShortTermFeatures.feature_extraction(y, 8000, 0.050 *8000, 0.025 * 8000)
    # features=np.reshape(np.vstack((F)),(440,68))

    # from sklearn.feature_extraction import DictVectorizer
    # vec = DictVectorizer()
    #
    # vec.fit_transform(features).toarray()
    #features.size
    if (fold_no != test_fold):
        x_train0.append(mfccs0)  # features
        y_train0.append(label)
        x_train1.append(mfccs1)  # features
        y_train1.append(label)
        x_train2.append(mfccs2)  # features
        y_train2.append(label)
        x_train3.append(mfccs3)  # features
        y_train3.append(label)
        x_train4.append(mfccs4)  # features
        y_train4.append(label)
    else:
        x_test0.append(mfccs0)  # features
        y_test0.append(label)
        x_test1.append(mfccs1)  # features
        y_test1.append(label)
        x_test2.append(mfccs2)  # features
        y_test2.append(label)
        x_test3.append(mfccs3)  # features
        y_test3.append(label)
        x_test4.append(mfccs4)  # features
        y_test4.append(label)


#x_train = x_train.flatten()
# Converting the lists into numpy arrays
x_train0 = np.array(x_train0)
x_test0 = np.array(x_test0)
y_train0 = np.array(y_train0)
y_test0 = np.array(y_test0)

x_train1 = np.array(x_train1)
x_test1 = np.array(x_test1)
y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

x_train2 = np.array(x_train2)
x_test2 = np.array(x_test2)
y_train2 = np.array(y_train2)
y_test2 = np.array(y_test2)

x_train3 = np.array(x_train3)
x_test3 = np.array(x_test3)
y_train3 = np.array(y_train3)
y_test3 = np.array(y_test3)

x_train4 = np.array(x_train4)
x_test4 = np.array(x_test4)
y_train4 = np.array(y_train4)
y_test4 = np.array(y_test4)

# x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# x_train_2d.shape,x_test_2d.shape


# Load feature vectors in file .csv
path = "/Users/PilvioSol/Desktop/PROVA0305/" #"C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
np.savetxt(path + '10_x_train0.csv', x_train0, delimiter=',')
np.savetxt(path + '10_x_test0.csv', x_test0, delimiter=',')
np.savetxt(path + '10_y_train0.csv', y_train0, delimiter=',')
np.savetxt(path + '10_y_test0.csv', y_test0, delimiter=',')

np.savetxt(path + '10_x_train1.csv', x_train1, delimiter=',')
np.savetxt(path + '10_x_test1.csv', x_test1, delimiter=',')
np.savetxt(path + '10_y_train1.csv', y_train1, delimiter=',')
np.savetxt(path + '10_y_test1.csv', y_test1, delimiter=',')

np.savetxt(path + '10_x_train2.csv', x_train2, delimiter=',')
np.savetxt(path + '10_x_test2.csv', x_test2, delimiter=',')
np.savetxt(path + '10_y_train2.csv', y_train2, delimiter=',')
np.savetxt(path + '10_y_test2.csv', y_test2, delimiter=',')

np.savetxt(path + '10_x_train3.csv', x_train3, delimiter=',')
np.savetxt(path + '10_x_test3.csv', x_test3, delimiter=',')
np.savetxt(path + '10_y_train3.csv', y_train3, delimiter=',')
np.savetxt(path + '10_y_test3.csv', y_test3, delimiter=',')

np.savetxt(path + '10_x_train4.csv', x_train4, delimiter=',')
np.savetxt(path + '10_x_test4.csv', x_test4, delimiter=',')
np.savetxt(path + '10_y_train4.csv', y_train4, delimiter=',')
np.savetxt(path + '10_y_test4.csv', y_test4, delimiter=',')


