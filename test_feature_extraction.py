import os
from util import extract_feature, read_image, write_feature
from tqdm import tqdm

import numpy as np

import tensorflow as tf

STACK_SIZE = 128
DATA_ORIGIN = 'data/origin'
DATA_FEATURE = 'data/feature'
DATA_FEATURE_512 = 'data/feature_512'

feature_data = []
feature_512_data = []

# for train_type in os.listdir(DATA_FEATURE):
#     for sample_id in os.listdir(DATA_FEATURE + '/' + train_type):
#         for subtype in os.listdir(DATA_FEATURE + '/' + train_type + '/' + sample_id):
#             for image in os.listdir(DATA_FEATURE + '/' + train_type + '/' + sample_id + '/' + subtype):
#                 string_data = DATA_FEATURE + '/' + train_type + '/' + sample_id + '/' + subtype + '/' + image
#                 feature_data.append(string_data.split('.')[0])

for train_type in os.listdir(DATA_FEATURE_512):
    for sample_id in os.listdir(DATA_FEATURE_512 + '/' + train_type):
        for subtype in os.listdir(DATA_FEATURE_512 + '/' + train_type + '/' + sample_id):
            for image in os.listdir(DATA_FEATURE_512 + '/' + train_type + '/' + sample_id + '/' + subtype):
                string_data = DATA_FEATURE_512 + '/' + train_type + '/' + sample_id + '/' + subtype + '/' + image
                feature_512_data.append(string_data.split('.')[0])

max_total = 0
max_link = 0
for train_type in os.listdir(DATA_ORIGIN):
    for sample_id in os.listdir(DATA_ORIGIN + '/' + train_type):
        total = 0
        for subtype in ['FLAIR', 'T1w', 'T2w']:
            if os.path.isdir(DATA_ORIGIN + '/' + train_type + '/' + sample_id + '/' + subtype):
                for image in os.listdir(DATA_ORIGIN + '/' + train_type + '/' + sample_id + '/' + subtype):
                    string_data = DATA_ORIGIN + '/' + train_type + '/' + sample_id + '/' + subtype + '/' + image
                    # if string_data.split('.')[0].replace(DATA_ORIGIN, DATA_FEATURE) not in feature_data:
                    #     raise ValueError('ERROR')
                    if string_data.split('.')[0].replace(DATA_ORIGIN, DATA_FEATURE_512) not in feature_512_data:
                        # raise ValueError('ERROR')
                        print(string_data)
                    if 'train' in string_data:
                        total += 1
        if total > max_total:
            max_total = total
            max_link = DATA_ORIGIN + '/' + train_type + '/' + sample_id

print('total ', max_total, max_link)

print('Everything OK')