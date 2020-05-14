import numpy as np
import matplotlib.pyplot as plt

# Load accuracies & confusion matrices
path = "C:/Users/Paolo De Santis/Desktop/Repository - CMLS - HM1/Pdf & Results/"

SVC_accuracies = np.genfromtxt(path + 'SVC_accuracies.csv', delimiter=',', dtype=float)
KNN_accuracies = np.genfromtxt(path + 'KNN_accuracies.csv', delimiter=',', dtype=float)
SVC_conf_mat = np.genfromtxt(path + 'SVC_conf_mat.csv', delimiter=',', dtype=float)
KNN_conf_mat = np.genfromtxt(path + 'KNN_conf_mat.csv', delimiter=',', dtype=float)

#BOXPLOT Accuiracies

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_title('SVC 10-Cross-Validation Accuracies Boxplot')
ax1.boxplot(SVC_accuracies)
ax2.set_title('KNN 10-Cross-Validation Accuracies Boxplot')
ax2.boxplot(KNN_accuracies)

plt.show()

# Average Confusion Matrices
m = 0
n = 0
big_mat = []
mat = []
a = []
for i in KNN_conf_mat: # SVC/KNN
    m += 1
    n += 1
    a.append(i)
    if m == 10:
        arr = np.array(a)
        mat.append(arr)
        m = 0
        a.clear()

    if n == 100:
        mat_arr = np.array(mat)
        big_mat.append(mat_arr)
        n = 0
        mat.clear()

conf_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
for i in big_mat:
    # iterate through rows
    for j in range(len(conf_mat)):
        # iterate through columns
        for k in range(len(conf_mat[0])):
            conf_mat[j][k] = conf_mat[j][k] + i[j][k]

print(conf_mat/10)


