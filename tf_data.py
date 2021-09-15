from retrain_model import feature_extractor
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf

from util import augment_data_split, image_generator, image_generator_image, image_generator_image_3d, sequence_generator

DATA_ORIGIN_TRAIN = 'data/origin/train'
DATA_FEATURE_TRAIN = 'data/feature_512/train'

fig, axs = plt.subplots(1, 2)
fig.suptitle('Vertically stacked subplots')

class AutoEncoderDataset(tf.data.Dataset):
    def __new__(self, folder, samples, batch_size):
        data = tf.data.Dataset.from_generator(
            image_generator,
            output_signature = (
                tf.TensorSpec(shape = (batch_size, 100352), dtype = tf.float32),
                tf.TensorSpec(shape = (batch_size, 32, 32, 1), dtype = tf.float32),
            ),
            args=(folder, samples, batch_size)
        )
        return data

class AutoEncoderImageDataset(tf.data.Dataset):
    def __new__(self, folder, samples, batch_size):
        data = tf.data.Dataset.from_generator(
            image_generator_image,
            output_signature = (
                tf.TensorSpec(shape = (batch_size, 240, 240, 1), dtype = tf.float32),
                tf.TensorSpec(shape = (batch_size, 32, 32, 1), dtype = tf.float32),
            ),
            args=(folder, samples, batch_size)
        )
        return data

class AutoEncoderImage3DDataset(tf.data.Dataset):
    def __new__(self, folder, samples, batch_size):
        data = tf.data.Dataset.from_generator(
            image_generator_image_3d,
            output_signature = (
                tf.TensorSpec(shape = (batch_size, 50, 240, 240, 4), dtype = tf.float32),
                tf.TensorSpec(shape = (batch_size, 50, 240, 240, 4), dtype = tf.float32),
            ),
            args=(folder, samples, batch_size)
        )
        return data

class TransformerDataset(tf.data.Dataset):
    def __new__(self, folder, samples, labels, batch_size, isTest=False):
        data = tf.data.Dataset.from_generator(
            sequence_generator,
            output_signature = (
                tf.TensorSpec(shape = (None, 120, 512), dtype = tf.float32),
                tf.TensorSpec(shape = (None), dtype = tf.float32),
            ),
            args=(folder, samples, labels, batch_size, isTest)
        )
        return data

if __name__ == "__main__":
    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for image, gray in dataset:
                print(image.shape, gray.shape)
                # Performing a training step
        print("Execution time:", time.perf_counter() - start_time)

    with open('pickle/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
        f.close()
    with open('pickle/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
        f.close()
    with open('pickle/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
        f.close()
    with open('pickle/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
        f.close()

    batch_size = 32
    # test_data = AutoEncoderDataset(DATA_ORIGIN_TRAIN, X_train, batch_size).prefetch(tf.data.AUTOTUNE)
        
    DATA_AUGMENT_TRAIN = '/media/sang/Samsung/data_augement/train'
    X_test, _ = augment_data_split(X_test, y_test)
    test_data = AutoEncoderImage3DDataset(DATA_AUGMENT_TRAIN, X_test, batch_size).prefetch(tf.data.AUTOTUNE)
        
    benchmark(test_data)