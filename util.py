import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        link = link.replace('origin', 'feature')
        link = link.replace('.png', '.npy')
        folder = '/'.join(link.split('/')[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(link, image)

def write_feature_512(images, links):
    for image, link in zip(images, links):
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

def padding_data(data, max_len=400):
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

def extract_feature_512(model, image_stacks, link_stacks):
    print('image_stacks.shape ', image_stacks.shape)
    features = model.feature_extract(image_stacks)
    write_feature_512(features, link_stacks)

def generate_image(model, images, epoch, isFull=True):
    predictions = None
    if isFull:
        predictions = model(images, training=False)
    else:
        predictions = model.image_generate(images)
    for idx, image in enumerate(predictions):
        image = np.reshape(image, (32, 32))
        image = (image * 127.5) + 127.5
        image = image.astype(np.int32)
        cv2.imwrite(f'images/image_{epoch}_{idx}.png', image)

def get_random_sequence(folder_link, type, reduce=0.8):
    sequence = np.array([])
    folder = os.listdir(f"{folder_link}/{type.decode('utf8')}")
    index = int(len(folder) * reduce)
    random_index = random.randint(index, len(folder))
    for image in random_data(folder)[:random_index]:
        image = np.load(f"{folder_link}/{type.decode('utf8')}/{image}")
        image = np.expand_dims(image, 0)
        if len(sequence) == 0:
            sequence = image
        else:
            sequence = np.concatenate([sequence, image], 0)
    length = np.array([len(sequence)])
    sequence = padding_data(sequence)
    sequence = np.expand_dims(sequence, 0)
    return sequence, length

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

def sequence_generator(data_folder, sample_list, labels_list, type, batch_size=128):
    sequences = np.array([])
    labels = np.array([])
    lengths = np.array([])
    # for i in range(10):
    #     yield (np.ones((batch_size, 224, 224, 3)), np.ones((batch_size, 224, 224, 1)))
    sample_list, labels = random_datas(sample_list, labels_list)
    with tqdm(total=len(sample_list)) as pbar:
        for idx, (image, label) in enumerate(zip(sample_list, labels)):
            try:
                # Reading data (line, record) from the file
                folder_link = f"{data_folder.decode('utf8')}/{image.decode('utf8')}"
                # folder_link = f'{data_folder}/{image}'
                label = np.array([[label]])

                sequence, length = get_random_sequence(folder_link, type)

                if sequences.shape[0] >= batch_size or idx >= len(sample_list) - 1:
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