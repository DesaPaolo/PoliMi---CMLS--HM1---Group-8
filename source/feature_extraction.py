import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.signal import butter, lfilter, freqz

# DATA COLLECTION

# path of UrbanSound8K 10 folds
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K"  # /Users/PilvioSol/Desktop/UrbanSound8K

# Read metadata file
df = pd.read_csv(path + "/metadata/UrbanSound8K.csv")


# Reorganize df (choose folds in if) ---> if you want 10 folds comment this for
# data = []
#
# for i in tqdm(range(len(df))):
#     fold_no = str(df.iloc[i]["fold"])
#     if fold_no == '1' or fold_no == '2' or fold_no == '3':
#         data.append(df.iloc[i])
#
# df = pd.DataFrame(data)


# PRE-PROCESSING OF AUDIO DATA

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

x_train = []  # train feature coeffs vector
x_test = []  # test feature coeffs vector
y_train = []  # train labels
y_test = []  # test labels

test_fold = '9'
num_coeff = 40


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_coeff)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


for i in tqdm(range(len(df))):

    fold_no = str(df.iloc[i]["fold"])
    file = df.iloc[i]["slice_file_name"]
    label = df.iloc[i]["classID"]
    filename = path + "/audio/fold" + fold_no + "/" + file

    # MFFCs
    features = extract_features(filename)

    if fold_no != test_fold:
        x_train.append(features)
        y_train.append(label)

    else:
        x_test.append(features)
        y_test.append(label)

# Convert lists into numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# LOAD FEATURE VECTORS IN .CSV FILES

path = "C:/Users/Paolo De Santis/Desktop/Repository - CMLS - HM1/Feature Vectors Archive/Test fold 8/"

np.savetxt(path + 'x_train.csv', x_train, delimiter=',')
np.savetxt(path + 'x_test.csv', x_test, delimiter=',')
np.savetxt(path + 'y_train.csv', y_train, delimiter=',')
np.savetxt(path + 'y_test.csv', y_test, delimiter=',')
