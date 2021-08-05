import os
import pickle
from util import extract_feature, extract_feature_512, read_image

import numpy as np

import tensorflow as tf

from models import AutoEncoder

from tqdm import tqdm

with open('pickle/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    f.close()

STACK_SIZE = 128
DATA_ORIGIN = 'data/origin/train'

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

with tqdm(total=len(X_test)) as pbar:
    for sub_folder in X_test:
        for type_image in os.listdir(DATA_ORIGIN + '/' + sub_folder):
            for image in os.listdir(DATA_ORIGIN  + '/' + sub_folder + '/' + type_image):
                image_path = DATA_ORIGIN + '/' + sub_folder + '/' + type_image + '/' + image
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