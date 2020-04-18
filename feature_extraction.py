import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

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
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K"  # /Users/PilvioSol/Desktop/UrbanSound8K

# Read metadata file
df = pd.read_csv(path + "/metadata/UrbanSound8K.csv")

# # Reorganize df (choose folds in if) ---> if you want 10 folds comment this for
# data = []
# for i in tqdm(range(len(df))):
#     fold_no = str(df.iloc[i]["fold"])
#     if (fold_no == '1' or fold_no == '2'):  # or fold_no == '3'):
#         data.append(df.iloc[i])
#
# df = pd.DataFrame(data)

# FEATURE EXTRACTION
x_train = []  # feature coeffs
x_test = []  # feature coeffs
y_train = []  # labels
y_test = []  # labels (desidered outputs)

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]]

test_fold = '10'

for i in tqdm(range(len(df))):
    fold_no = str(df.iloc[i]["fold"])
    file = df.iloc[i]["slice_file_name"]
    label = df.iloc[i]["classID"]
    filename = path + "/audio/fold" + fold_no + "/" + file
    y, sr = librosa.load(filename)  # convert to mono
    y = parse_audio(y)

    # MFFCs
    mfccs = np.mean(librosa.feature.mfcc(y, sr).T, axis=0)  # n_mfcc = 20 (default value)

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

# Load feature vectors in file .csv
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
np.savetxt(path + '1-9folds_mfcc_n_coeff=20_x_train.csv', x_train, delimiter=',')
np.savetxt(path + 'fold10_mfcc_n_coeff=20_x_test.csv', x_test, delimiter=',')
np.savetxt(path + '1-9folds_y_train.csv', y_train, delimiter=',')
np.savetxt(path + 'fold10_y_test.csv', y_test, delimiter=',')
