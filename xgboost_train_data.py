import os
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

DATA_FEATURE_TRAIN = 'data/feature_512000/train'
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

X_train_folder, y_train_origin = augment_data_split(X_train, y_train, 1)
X_test_folder, y_test_origin = augment_data_split(X_test, y_test, 1)

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

    X_train = None
    y_train = None

    for idx in train_index:
        sub_folder = X_folder[idx]
        # sub_folder = sub_folder.split('_')[0]
        # for i in range(14):
        #     sub_folder = sub_folder + '_' + str(i)
        label = y_origin[idx]
        if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
            feature = None
            for type_image in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
                if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                    for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder + '/' + type_image):
                        image_path = DATA_FEATURE_TEST + '/' + '/' + sub_folder + '/' + type_image + '/' + image
                        image = np.load(image_path)
                        image = np.expand_dims(image, 0)
                        label = np.array([int(label)])

                        if feature is None:
                            feature = image
                        else:
                            feature = np.concatenate([feature, image], 1)
            if X_train is None:
                X_train = feature
                y_train = label
            else:
                X_train = np.concatenate([X_train, feature], 0)
                y_train = np.concatenate([y_train, label], 0)

    param_grid = { 
        'n_estimators': [10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [1,2,3,4,5],
        'criterion' :['gini', 'entropy']
    }
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    # model = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)

    ##############################################################################
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_train)
    predictions = [np.argmax(value) for value in y_pred]

    accuracy = roc_auc_score(y_train, predictions)
    train_score.append(accuracy * 100.0)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    total = 0
    total_true = 0
    for idx in test_index:
        sub_folder = X_folder[idx]
        label = y_origin[idx]
        if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder):
            feature = None
            count = 0
            for type_image in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
                if os.path.isdir(DATA_FEATURE_TRAIN + '/' + sub_folder + '/' + type_image):
                    for image in os.listdir(DATA_FEATURE_TRAIN + '/' + '/' + sub_folder + '/' + type_image):
                        image_path = DATA_FEATURE_TRAIN + '/' + '/' + sub_folder + '/' + type_image + '/' + image
                        image = np.load(image_path)
                        image = np.expand_dims(image, 0)
                        
                        if feature is None:
                            feature = image
                        else:
                            feature = np.concatenate([feature, image], 1)

            y_pred = model.predict_proba(feature)     
            if np.argmax(y_pred[0]) == label:
                total_true += 1
            total += 1

    accuracy = total_true / total * 100.0
    test_score.append(accuracy)

    print("Test Accuracy: %.2f%%" % (accuracy))

print('score ', np.mean(np.array(test_score)))