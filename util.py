import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random
import numpy as np

import tensorflow as tf

from retrain_model import feature_extractor

def create_padding_mask(seq):
    seq = tf.cast(seq, tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]

def write_feature(images, links):
    for image, link in zip(images, links):
        link = link.replace('data/agument', '/media/sang/b2f3b20a-1ca2-4225-990d-8459f5a80739/feature')
        link = link.replace('.png', '.npy')
        folder = '/'.join(link.split('/')[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(link, image)

def write_feature_512(images, links, isOrigin=False):
    for image, link in zip(images, links):
        if isOrigin == True:
            # link = link.replace('origin', 'feature_512_origin')
            link = link.replace('data/origin/train', 'data/test_origin')
            link = link.replace('.png', '.npy')
        else:
            link = link.replace('feature', 'feature_512')
        folder = '/'.join(link.split('/')[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(link, image)

def random_data(data):
    random.shuffle(data)
    return data

def random_datas(data, labels):
    sshuffler = np.random.permutation(len(data))
    data_shuffled = data[sshuffler]
    labels_shuffled = labels[sshuffler]
    return data_shuffled, labels_shuffled

def random_datas_three(data, lengths, labels):
    sshuffler = np.random.permutation(len(data))
    data_shuffled = data[sshuffler]
    lengths_shuffled = lengths[sshuffler]
    labels_shuffled = labels[sshuffler]
    return data_shuffled, lengths_shuffled, labels_shuffled

def padding_data(data, max_len=1000):
    while len(data) < max_len:
        data = np.concatenate([data, np.zeros((1, 512))], 0)
    return data

def clip_data(data, lengths):
    max_length = np.max(lengths)

    results = np.array([])
    for feature in data:
        clip_feature = feature[:max_length]
        clip_feature = np.expand_dims(clip_feature, 0)
        if results.shape[0] == 0:
            results = clip_feature
        else:
            results = np.concatenate([results, clip_feature], 0)

    masks = np.array([])
    for length in lengths:
        mask = [0] * length
        while len(mask) < max_length:
            mask.append(1)
        mask = np.expand_dims(mask, 0)
        if masks.shape[0] == 0:
            masks = mask
        else:
            masks = np.concatenate([masks, mask], 0)

    return results, create_padding_mask(masks)

def read_image(image_path):
    origin_image = cv2.imread(image_path)
    origin_image = cv2.resize(origin_image, (224, 224))
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, 0)
    return image

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

def extract_feature_512(model, image_stacks, link_stacks, isOrigin=False):
    features = model.feature_extract(image_stacks)
    write_feature_512(features, link_stacks, isOrigin)

def generate_image(model, images, links, epoch, isFull=True):
    predictions = None
    if isFull:
        predictions = model(images, training=False)
    else:
        predictions = model.image_generate(images)
    for idx, (image, link) in enumerate(zip(predictions, links)):
        image = np.reshape(image, (32, 32))
        image = (image * 127.5) + 127.5
        image = image.astype(np.int32)
        link = link.replace('test_origin', 'image_test')
        file_paths = link.split('/')
        folder_name = '/'.join(file_paths[:-1])
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        cv2.imwrite(f'{folder_name}/image_{epoch}_{idx}.png', image)

def get_random_sequence(folder_link, reduce=0.8, isTest=False):
    sequences = np.array([])
    for type in ['FLAIR', 'T1w', 'T2w']:
        if os.path.isdir(f"{folder_link}/{type}"):
            sequence = np.array([])
            folder = os.listdir(f"{folder_link}/{type}")
            if not isTest and len(folder) > 50:
                index = round(len(folder) * reduce)
                random_index = random.randint(index, len(folder))
                folder = random_data(folder)[:random_index]
            for image in folder:
                image = np.load(f"{folder_link}/{type}/{image}")
                image = np.expand_dims(image, 0)
                if len(sequence) == 0:
                    sequence = image
                else:
                    sequence = np.concatenate([sequence, image], 0)
            if sequences.shape[0] == 0:
                sequences = sequence
            else:
                sequences = np.concatenate([sequences, sequence], 0)
        
    length = np.array([len(sequences)])
    sequences = padding_data(sequences)
    sequences = np.expand_dims(sequences, 0)
    return sequences, length

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

def sequence_generator(data_folder, sample_list, labels_list, type, batch_size=128, isTest=False):
    sequences = np.array([])
    labels = np.array([])
    lengths = np.array([])
    # for i in range(10):
    #     yield (np.ones((batch_size, 224, 224, 3)), np.ones((batch_size, 224, 224, 1)))
    # if isTest:
    #     sample_list = np.tile(sample_list, 10)
    #     labels_list = np.tile(labels_list, 10)
    # else:
    #     sample_list = np.tile(sample_list, 20)
    #     labels_list = np.tile(labels_list, 20)
    if not isTest:
        sample_list, labels_list = random_datas(sample_list, labels_list)

    with tqdm(total=len(sample_list)) as pbar:
        for idx, (image, label) in enumerate(zip(sample_list, labels_list)):
            try:
                # Reading data (line, record) from the file
                folder_link = f"{data_folder.decode('utf8')}/{image.decode('utf8')}"
                # folder_link = f'{data_folder}/{image}'
                label = np.array([int(label)])

                sequence, length = get_random_sequence(folder_link, 0.9, isTest)

                if sequences.shape[0] >= batch_size or idx >= len(sample_list) - 1:
                    if not isTest:
                        sequences, lengths, labels = random_datas_three(sequences, lengths, labels)
                    sequences, masks = clip_data(sequences, lengths)
                    yield sequences, masks, labels
                    sequences = sequence
                    labels = label
                    lengths = length
                else:
                    if sequences.shape[0] == 0:
                        sequences = sequence
                        labels = label
                        lengths = length
                    else:
                        sequences = np.concatenate([sequences, sequence], 0)
                        labels = np.concatenate([labels, label], 0)
                        lengths = np.concatenate([lengths, length], 0)
            except:
                pass
            
            pbar.update(1)

def strong_aug(object_type):
    return A.Compose([
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.2], contrast_limit=0, p=0.5),
            A.HorizontalFlip(p=0.5)
        ],
        additional_targets=object_type
    )

def augment_data(folder, feature, n_generated_samples):
    for type_name in os.listdir(folder + '/' + feature):
        for idx in range(n_generated_samples):

            aug_input = {}
            image_name = []
            object_type = {}
            file_path_list = {}

            for id, image in enumerate(os.listdir(folder + '/' + feature + '/' + type_name)):
                # load the image
                file_path = folder + '/' + feature + '/' + type_name + '/' + image
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if id == 0:
                    aug_input['image'] = image
                    file_path_list['image'] = file_path
                    image_name.append('image')
                else:
                    aug_input['image' + str(id - 1)] = image
                    file_path_list['image' + str(id - 1)] = file_path
                    object_type['image' + str(id - 1)] = 'image'
                    image_name.append('image' + str(id - 1))

            aug = strong_aug(object_type)
            augmented_data = aug(**aug_input)
            
            for name in image_name:
                image = augmented_data[name]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                file_path = file_path_list[name]
                file_path = file_path.replace('origin', 'agument')
                file_paths = file_path.split('/')
                file_paths[3] = f'{feature}_{str(idx)}'
                file_path = '/'.join(file_paths)
                folder_name = '/'.join(file_paths[:-1])
                if not os.path.isdir(folder_name):
                    os.makedirs(folder_name)
                cv2.imwrite(file_path, image)

def augment_data(X_data, y_data):
    new_x = []
    new_y = []
    
    for x, y in zip(X_data, y_data):
        for idx in range(10):
            new_x.append(x + '_' + str(idx))
            new_y.append(y)

    return new_x, new_y