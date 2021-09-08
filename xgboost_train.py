import os
from tqdm import tqdm
from util import augment_data_split, random_datas
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

DATA_FEATURE_TRAIN = 'data/feature_512/train'
DATA_FEATURE_TEST = 'data/feature_512/train'

with open('pickle/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
    f.close()

with open('pickle/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    f.close()

with open('pickle/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
    f.close()

with open('pickle/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
    f.close()

X = None
y = None

X_train_folder, y_train = augment_data_split(X_train, y_train)
X_test_folder, y_test = augment_data_split(X_test, y_test)

label_true = 0
label_false = 0

with tqdm(total=len(X_train_folder + X_test_folder)) as pbar:
    for sub_folder, label in zip(X_train_folder + X_test_folder, y_train + y_test):
        if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
            for type_image in ['FLAIR']:
                if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                    for image in os.listdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                        image_path = DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image + '/' + image
                        image = np.load(image_path)
                        image = np.expand_dims(image, 0)
                        label = np.array([int(label)])
                        if X is None:
                            X = image
                            y = label
                        else:
                            X = np.concatenate([X, image], 0)
                            y = np.concatenate([y, label], 0)
        pbar.update(1)

# fit model no training data
model = XGBClassifier(objective='binary:logistic', eval_metric='auc')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# # summarize performance
# print('scores ', scores)
# print('Mean ROC AUC: %.5f' % np.mean(scores))

# model.fit(x_feature_train, label_train)

# y_pred = model.predict(x_feature_train)
# predictions = [round(value) for value in y_pred]

# accuracy = roc_auc_score(label_train, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# y_pred = model.predict(x_feature_test)
# predictions = [round(value) for value in y_pred]

# accuracy = roc_auc_score(label_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

train_score = []
test_score = []
for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
    print('=========================================================')
    print("fold:", fold, len(train_index), len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = XGBClassifier(objective='binary:logistic', eval_metric='auc')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    predictions = [round(value) for value in y_pred]

    accuracy = roc_auc_score(y_train, predictions)
    train_score.append(accuracy * 100.0)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = roc_auc_score(y_test, predictions)
    test_score.append(accuracy * 100.0)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    X_test_folder_train = [(X_train_folder + X_test_folder)[id] for id in train_index]
    X_test_folder_test = [(X_train_folder + X_test_folder)[id] for id in test_index]

print(np.mean(train_score), np.mean(test_score))

total = 0
total_true = 0
for sub_folder, label in zip(X_test_folder_train, y_train):
    if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
        for type_image in ['FLAIR']:
            if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder + '/' + type_image):
                    image_path = DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + type_image + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    y_pred = model.predict(image)
                    if round(y_pred[0]) == label:
                        total_true += 1
                    total += 1

print(total, total_true, (total_true / total * 100))

total = 0
total_true = 0
for sub_folder, label in zip(X_test_folder_test, y_test):
    if os.path.isdir(DATA_FEATURE_TEST + '/' + sub_folder):
        for type_image in ['FLAIR']:
            if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                for image in os.listdir(DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + type_image):
                    image_path = DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + type_image + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    y_pred = model.predict(image)
                    if round(y_pred[0]) == label:
                        total_true += 1
                    total += 1

print(total, total_true, (total_true / total * 100))
