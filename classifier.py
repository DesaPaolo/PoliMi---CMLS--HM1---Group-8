import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# LOAD FEATURE VECTOR FROM .CSV FILES
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/" #"C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
x_train = np.genfromtxt(path + '1-6,8-10folds_mfcc_n_coeff=40_x_train.csv', delimiter=',')
x_test = np.genfromtxt(path + 'fold7_mfcc_n_coeff=40_x_test.csv', delimiter=',')
y_train = np.genfromtxt(path + '1-6,8-10folds_y_train.csv', delimiter=',')
y_test = np.genfromtxt(path + 'fold7_y_test.csv', delimiter=',')

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# CLASSIFIER

# GRID SEARCH CV X KNN
# grid_params = {
#     'n_neighbors': [3, 5, 7, 9, 11, 15],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
# Accuracy: 0.52

# GRID SEARCH CV x LINEAR SVC
grid_params_1 = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
# Accuracy:
# 0.6129032258064516 MFCCs 20 coeffs test fold 10
# 0.522673031026253 MFCCs 13 coeffs test fold 7
# 0.5894988066825776 MFCCs 40 coeff test fold 7

# grid_params_2 = [{
#     'kernel': ['rbf'],
#     'gamma': [1e-3, 1e-4],
#     'C': [1, 10, 100, 1000]
# }, {
#     'kernel': ['linear'],
#     'C': [1, 10, 100]
# }]
# Accuracy: 0.60

model = GridSearchCV(SVC(), grid_params_1, refit=True, verbose=2, cv=5);
# model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)


model.fit(x_train_scaled, y_train)
y_predict = model.predict(x_test_scaled)

# Evaluation
accuracy = model.score(x_test_scaled, y_test)  # x_test_scaled/x_test
print("Accuracy:")
print(accuracy)
print(confusion_matrix(y_test, y_predict))

