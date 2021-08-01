from retrain_model import feature_extractor
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf

from util import image_generator

DATA_ORIGIN_TRAIN = 'data/origin/train'

fig, axs = plt.subplots(1, 2)
fig.suptitle('Vertically stacked subplots')

class AutoEncoderDataset(tf.data.Dataset):
    def __new__(self, folder, samples, batch_size):
        data = tf.data.Dataset.from_generator(
            image_generator,
            output_signature = (
                tf.TensorSpec(shape = (batch_size, 100352), dtype = tf.float32),
                tf.TensorSpec(shape = (batch_size, 64, 64, 1), dtype = tf.float32),
            ),
            args=(folder, samples, batch_size)
        )
        return data

if __name__ == "__main__":
    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for image, gray in dataset:
                print(image.shape, np.sum(image))
                print(gray.shape, np.sum(gray))
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
    test_data = AutoEncoderDataset(DATA_ORIGIN_TRAIN, X_train, batch_size).prefetch(tf.data.AUTOTUNE)
        
    benchmark(test_data)