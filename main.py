
import os

# Data managing

folds = ['1','2','3','4','5','6','7','8','9','10']

for fold in folds:
    path, dirs, files = next(os.walk('C:/Users/Paolo De Santis/Desktop/UrbanSound/UrbanSound8K/audio/fold' + fold))
    for file in files:
        if file.endswith('.wav'):
            print(file)






