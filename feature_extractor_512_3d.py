
import os
import cv2
from util import extract_feature_512, padding_image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models import AutoEncoder, AutoEncoderFull, AutoEncoderFull3D

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

STACK_SIZE = 10
DATA_FEATURE = '/media/sang/Samsung/data_augement_new'
DATA_FEATURE_512 = 'data/feature_512000'

model = AutoEncoderFull3D()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

checkpoint_path = 'weights/autoencoder_full_3d_FLAIR_gray'

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
    for data_folder in ['test', 'train']:
        for sub_folder in os.listdir(DATA_FEATURE + '/' + data_folder):
            for type_image in ['FLAIR']:
                if os.path.isdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image):
                    image3d = None 
                    folder = os.listdir(DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image)
                    folder.sort(key=lambda x: int(x.replace('_', '-').replace('.', '-').split('-')[-2]))

                    start = len(folder) // 2 - 20
                    end = len(folder) // 2 + 20
                    
                    if start < 0:
                        start = 0
                        end += 40

                    for image in folder[start:end]:
                        image_path = DATA_FEATURE + '/' + data_folder + '/' + sub_folder + '/' + type_image + '/' + image
                        image = cv2.imread(image_path, 0)
                        image = np.expand_dims(image, -1)
                        image = np.expand_dims(image, 0)
                        if image3d is None:
                            image3d = image
                        else:
                            image3d = np.concatenate([image3d, image], 0)
                        
                    image3d = padding_image(image3d, 240, 1, 40)
                    image3d = np.expand_dims(image3d, 0)

                    if image_stacks is None:
                        image_stacks = image3d
                        link_stacks = [image_path]
                    else:
                        if image_stacks.shape[0] >= STACK_SIZE:
                            image_stacks = image_stacks.astype(np.float32)
                            image_stacks = image_stacks / 255
                            print('image_stacks.shape ', image_stacks.shape, np.min(image_stacks), np.max(image_stacks), np.mean(image_stacks))
                            extract_feature_512(model, image_stacks, link_stacks, DATA_FEATURE, DATA_FEATURE_512, True)
                            image_stacks = image3d
                            link_stacks = [image_path]
                        else:
                            image_stacks = np.concatenate([image_stacks, image3d], 0)
                            link_stacks.append(image_path)

                pbar.update(1)

extract_feature_512(model, image_stacks, link_stacks, DATA_FEATURE, DATA_FEATURE_512, True)