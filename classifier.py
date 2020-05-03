import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, LinearSVC

# Load feature vector
path = "/Users/PilvioSol/Desktop/PROVA0305/" #"C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
x_train = np.genfromtxt(path + '10_x_train1.csv', delimiter=',')
x_test = np.genfromtxt(path + '10_x_test1.csv', delimiter=',')
y_train = np.genfromtxt(path + '10_y_train1.csv', delimiter=',')
y_test = np.genfromtxt(path + '10_y_test1.csv', delimiter=',')

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Classifier
#GRID SEARCH CV + KNN
# grid_params = {
#     'n_neighbors': [3, 5, 7, 9, 11, 15],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
# #model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
# Accuracy: 0.52
#
# #GRID SEARCH CV + LINEAR SVC
# grid_params_1 = {
#     'C': [0.1, 1, 10],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }
# # Accuracy: 0.6129032258064516
# #
# grid_params_2 = [{
#     'kernel': ['rbf'],
#     'gamma': [1e-3, 1e-4],
#     'C': [1, 10, 100, 1000]
# }, {
#     'kernel': ['linear'],
#     'C': [1, 10, 100]
# }]
# Accuracy: 0.60
#
# #model =  GridSearchCV(SVC(), grid_params_1, refit=True, verbose=2);
#
# model = MLPClassifier(alpha=0.5, max_iter=1000),
#
#
#
#
# #Linear SVC
# model = LinearSVC(random_state=42 , C=10)
#
# model.fit(x_train_scaled, y_train)
# y_predict = model.predict(x_test_scaled)
#
# # Evaluation
# accuracy = model.score(x_test_scaled, y_test)  # x_test_scaled/x_test
# print("Accuracy:")
# print(accuracy)
# print(confusion_matrix(y_test, y_predict))

