
import os
from util import extract_feature_512
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models import AutoEncoder

STACK_SIZE = 128
DATA_FEATURE = 'data/feature'
DATA_FEATURE_512 = 'data/feature_512'

model = AutoEncoder()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

checkpoint_path = 'weights/autoencoder_relu_leaky_relu'

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

data = os.listdir(DATA_FEATURE)
test_length = len(os.listdir(DATA_FEATURE + '/' + 'test'))
train_length = len(os.listdir(DATA_FEATURE + '/' + 'train'))

image_stacks = None
link_stacks = []

with tqdm(total=(test_length + train_length)) as pbar:
    for data_folder in data:
        for sub_folder in os.listdir(DATA_FEATURE + '/' + data_folder):
            for type_image in os.listdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder):
                for image in os.listdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image):
                    image_path = DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    if image_stacks is None:
                        image_stacks = image
                        link_stacks = [image_path]
                    else:
                        if image_stacks.shape[0] >= STACK_SIZE:
                            extract_feature_512(model, image_stacks, link_stacks)
                            image_stacks = None
                        else:
                            image_stacks = np.concatenate([image_stacks, image], 0)
                            link_stacks.append(image_path)
            pbar.update(1)