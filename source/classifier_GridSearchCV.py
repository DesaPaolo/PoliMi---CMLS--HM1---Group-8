import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import GridSearchCV
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


# Set the path to the full UrbanSound dataset
fulldatasetpath = 'C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K/audio/'

metadata = pd.read_csv('C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K/metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))

    class_label = row["classID"]
    data = extract_features(file_name)

    features.append([data, class_label])

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Random splitting (train_test_split() computes firstly a shuffle on the entire input dataset)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# CLASSIFIER

# SVC Parameters
# grid_params_svc = {
#     'C': [0.1, 1, 10],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }

# KNN Parameters
grid_params_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# GRID SEARCH

# svc = SVC()
knn = KNeighborsClassifier()
clf = GridSearchCV(estimator=knn, param_grid=grid_params_knn, cv=10)
clf.fit(x_train_scaled, y_train)

# Evaluation
print(f'Model Score: {clf.score(x_test_scaled, y_test)}')
y_predict = clf.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')


