import os
import joblib
from tqdm import tqdm
from util import augment_data_split, random_datas
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

DATA_FEATURE_TRAIN = 'data/feature_512000/train'
DATA_FEATURE_TEST = 'data/feature_512000/train'

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

X_train_folder, y_train_origin = X_train, y_train
X_test_folder, y_test_origin = X_test, y_test

label_true = 0
label_false = 0

X_folder = X_train_folder + X_test_folder
y_origin = y_train_origin + y_test_origin

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

train_score = []
test_score = []
for fold, (train_index, test_index) in enumerate(cv.split(X_folder, y_origin)):
    print('=========================================================')
    print("fold:", fold, len(train_index), len(test_index))

    X_train= None
    X_train_NO = None

    y_train = None
    y_train_NO = None

    for idx in train_index:
        sub_folder = X_folder[idx]
        label = y_origin[idx]
        if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
            if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
                for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder):
                    image_path = DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    label = np.array([int(label)])
                    if X_train is None:
                        X_train = image
                        y_train = label
                    else:
                        X_train = np.concatenate([X_train, image], 0)
                        y_train = np.concatenate([y_train, label], 0)

    for sub_folder in os.listdir(DATA_FEATURE_TRAIN):
        if 'BraTS20' in sub_folder:
            if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
                for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder):
                    image_path = DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    label = np.array([-1])
                    if X_train_NO is None:
                        X_train_NO = image
                        y_train_NO = label
                    else:
                        X_train_NO = np.concatenate([X_train_NO, image], 0)
                        y_train_NO = np.concatenate([y_train_NO, label], 0)

    svc = SVC(probability=True, gamma="auto")
    model = SelfTrainingClassifier(svc)
    
    X_train_mix = np.concatenate([X_train, X_train_NO], 0)
    y_train_mix = np.concatenate([y_train, y_train_NO], 0)
    X_train_mix, y_train_mix = random_datas(X_train_mix, y_train_mix)
    model.fit(X_train_mix, y_train_mix)

    y_pred = model.predict(X_train)
    accuracy = roc_auc_score(y_train, y_pred)
    train_score.append(accuracy * 100.0)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    total = 0
    total_true = 0
    for idx in test_index:
        sub_folder = X_folder[idx]
        label = y_origin[idx]
        for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder):
            image_path = DATA_FEATURE_TRAIN + '/' + '/' + sub_folder + '/' + image
            image = np.load(image_path)
            image = np.expand_dims(image, 0)
            y_pred = model.predict_proba(image)
                                                    
            if np.argmax(y_pred[0]) == label:
                total_true += 1
            total += 1

    accuracy = total_true / total * 100.0
    test_score.append(accuracy)

    print("Test Accuracy: %.2f%%" % (accuracy))

print('score ', np.mean(np.array(test_score)))