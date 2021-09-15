import os
import csv
import pickle

from sklearn.model_selection import train_test_split

DATA_ORIGIN = '/media/sang/Samsung/data_augement_new/train'

features = []
labels = []

for folder in os.listdir(DATA_ORIGIN):
    if folder not in ['00109', '00123', '00709']:
        features.append(folder)
        labels.append(0)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

with open('pickle/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
with open('pickle/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
    f.close()
with open('pickle/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
with open('pickle/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
    f.close()