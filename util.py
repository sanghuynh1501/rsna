import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import random
import numpy as np

import tensorflow as tf

from retrain_model import feature_extractor

def random_data(data):
    random.shuffle(data)
    return data

def random_datas(data, labels):
    sshuffler = np.random.permutation(len(data))
    data_shuffled = data[sshuffler]
    labels_shuffled = labels[sshuffler]
    return data_shuffled, labels_shuffled

def read_image(image_path, gayscale=False):
    origin_image = cv2.imread(image_path)
    origin_image = cv2.resize(origin_image, (64, 64))
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, 0)
    if not gayscale:
        return image
    else:
        gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, -1)
        gray = np.expand_dims(gray, 0)
        return image, gray

def read_image_numpy(image_path, gayscale=False):
    numpy_path = image_path.replace('origin', 'feature')
    numpy_path = numpy_path.replace('.png', '.npy')
    image = np.load(numpy_path)
    image = np.expand_dims(image, 0)
    if not gayscale:
        return image
    else:
        gray = cv2.imread(image_path, 0)
        gray = cv2.resize(gray, (32, 32))
        gray = np.expand_dims(gray, -1)
        gray = np.expand_dims(gray, 0)
        return image, gray

def extract_feature(image):
    feature = feature_extractor(image)
    feature = np.reshape(feature, (128, -1))
    return feature

def image_generator(data_folder, sample_list, batch_size=128):
    images = np.array([])
    grays = np.array([])
    # for i in range(10):
    #     yield (np.ones((batch_size, 224, 224, 3)), np.ones((batch_size, 224, 224, 1)))
    with tqdm(total=len(sample_list)) as pbar:
        for sample in random_data(sample_list):
            # Reading data (line, record) from the file
            folder_link = f"{data_folder.decode('utf8')}/{sample.decode('utf8')}"
            for sub_folder in os.listdir(folder_link):
                for image in os.listdir(f'{folder_link}/{sub_folder}'):
                    try:
                        image, gray = read_image_numpy(f'{folder_link}/{sub_folder}/{image}', True)
                        if images.shape[0] >= batch_size:
                            images, grays = random_datas(images, grays)
                            grays = grays.astype(np.float32)
                            grays = (grays - 127.5) / 127.5
                            yield images, grays
                            images = image
                            grays = gray
                        else:
                            if images.shape[0] == 0:
                                images = image
                                grays = gray
                            else:
                                images = np.concatenate([images, image], 0)
                                grays = np.concatenate([grays, gray], 0)
                    except:
                        pass
            pbar.update(1)