
import os
from util import extract_feature_512, generate_image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models import AutoEncoder

STACK_SIZE = 256
DATA_FEATURE = 'data/feature_512'

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

min = float('inf')
max = float('-inf')

dem = 0

with tqdm(total=(test_length + train_length)) as pbar:
    for data_folder in data:
        for sub_folder in os.listdir(DATA_FEATURE + '/' + data_folder):
            for type_image in os.listdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder):
                for image in os.listdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image):
                    image_path = DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image + '/' + image
                    image = np.load(image_path)
                    image = np.expand_dims(image, 0)
                    # print(np.min(image))
                    if np.min(image) < min:
                        min = np.min(image)
                    if np.max(image) > max:
                        max = np.max(image)
                    if image_stacks is None:
                        image_stacks = image
                        link_stacks = [image_path]
                    else:
                        if image_stacks.shape[0] >= STACK_SIZE:
                            generate_image(model, image_stacks[:5], dem, isFull=False)
                            image_stacks = None
                            dem += 1
                        else:
                            image_stacks = np.concatenate([image_stacks, image], 0)
                            link_stacks.append(image_path)
            pbar.update(1)

print('min ', min, max)