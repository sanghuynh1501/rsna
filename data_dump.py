import os
import csv
import pickle

from sklearn.model_selection import train_test_split

DATA_ORIGIN = 'data/feature/train'

def get_data_len(folders, typedt='FLAIR'):
    count = 0
    for folder in folders:
        if os.path.isdir(f'{DATA_ORIGIN}/{folder}/{typedt}'):
            count += len(os.listdir(f'{DATA_ORIGIN}/{folder}/{typedt}'))
    return count

features = []
labels = []

with open('data/train_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            features.append(row[0])
            labels.append(int(row[1]))

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