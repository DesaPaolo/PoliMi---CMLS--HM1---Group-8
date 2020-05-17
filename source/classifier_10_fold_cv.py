import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# # LOAD FEATURE VECTOR FROM .CSV FILES
path = "C:/Users/Paolo De Santis/Desktop/Repository - CMLS - HM1/Feature Vectors Archive/Test fold 8/"
x_train = np.genfromtxt(path + 'x_train.csv', delimiter=',')
x_test = np.genfromtxt(path + 'x_test.csv', delimiter=',')
y_train = np.genfromtxt(path + 'y_train.csv', delimiter=',')
y_test = np.genfromtxt(path + 'y_test.csv', delimiter=',')


# Scaling of feature vectors
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

# model = SVC()
model = KNeighborsClassifier()

# Fit with parameters grid - wrapping parameter
for g in tqdm(ParameterGrid(grid_params_knn)):
    model.set_params(**g)
    model.fit(x_train_scaled, y_train)  # x_train_scaled/x_train

# Evaluation
accuracy = model.score(x_test_scaled, y_test)  # x_test_scaled/x_test
print("Accuracy:")
print(accuracy)
y_predict = model.predict(x_test_scaled)  # x_test_scaled/x_test
print(confusion_matrix(y_test, y_predict))
