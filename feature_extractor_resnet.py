import os
from util import extract_feature, read_image, write_feature
from tqdm import tqdm

import numpy as np

import tensorflow as tf

STACK_SIZE = 128
DATA_ORIGIN = 'data/agument'

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
                            write_feature(features, link_stacks)
                            image_stacks = image
                            link_stacks = [image_path]
                        else:
                            image_stacks = np.concatenate([image_stacks, image], 0)
                            link_stacks.append(image_path)
            pbar.update(1)