import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load feature vector
path = "C:/Users/Paolo De Santis/Desktop/UrbanSound/Feature coeff csv/"
x_train = np.genfromtxt(path + '1-9folds_mfcc_n_coeff=20_x_train.csv', delimiter=',')
x_test = np.genfromtxt(path + 'fold10_mfcc_n_coeff=20_x_test.csv', delimiter=',')
y_train = np.genfromtxt(path + '1-9folds_y_train.csv', delimiter=',')
y_test = np.genfromtxt(path + 'fold10_y_test.csv', delimiter=',')


scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Classifier
grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)
y_predict = model.predict(x_test_scaled)

#Evaluation
accuracy = model.score(x_test_scaled, y_test)#x_test_normalized/x_test
print("Accuracy:")
print(accuracy)
print(confusion_matrix(y_test, y_predict))

