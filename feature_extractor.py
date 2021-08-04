import os
from util import extract_feature, read_image, extract_feature_512
from tqdm import tqdm

import numpy as np

import tensorflow as tf

from models import AutoEncoder

STACK_SIZE = 128
DATA_ORIGIN = 'data/origin'

model = AutoEncoder()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

checkpoint_path = 'weights/autoencoder_relu_leaky_relu'

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

image_stacks = None
link_stacks = []
data = os.listdir(DATA_ORIGIN)
test_length = len(os.listdir(DATA_ORIGIN + '/' + 'test'))
train_length = len(os.listdir(DATA_ORIGIN + '/' + 'train'))

with tqdm(total=(test_length + train_length)) as pbar:
    for data_folder in data:
        for sub_folder in os.listdir(DATA_ORIGIN + '/' + data_folder):
            for type_image in os.listdir(DATA_ORIGIN + '/' + data_folder + '/' + sub_folder):
                for image in os.listdir(DATA_ORIGIN + '/' + data_folder + '/' + sub_folder + '/' + type_image):
                    image_path = DATA_ORIGIN + '/' + data_folder + '/' + sub_folder + '/' + type_image + '/' + image
                    image = read_image(image_path)
                    if image_stacks is None:
                        image_stacks = image
                        link_stacks = [image_path]
                    else:
                        if image_stacks.shape[0] >= STACK_SIZE:
                            features = extract_feature(image_stacks)
                            # write_feature(features, link_stacks)
                            extract_feature_512(model, features, link_stacks, True)
                            image_stacks = image
                            link_stacks = [image_path]
                        else:
                            image_stacks = np.concatenate([image_stacks, image], 0)
                            link_stacks.append(image_path)
            pbar.update(1)