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

def write_feature(images, links, input_folder, output_folder):
    for image, link in zip(images, links):
        link = link.replace(input_folder, output_folder)
        link = link.replace('.png', '.npy')
        folder = '/'.join(link.split('/')[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(link, image)

def write_feature_512(images, links, input_folder, output_folder, isOrigin=False):
    for image, link in zip(images, links):
        link = link.replace(input_folder, output_folder)
        if isOrigin == True:
            link = link.replace('.png', '.npy')
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

def padding_image(data, size, dim, max_len=1000):
    if len(data) == max_len - 1:
        data = np.concatenate([data, np.zeros((1, size, size, dim))], 0)
    elif len(data) <= max_len - 2:
        haft_max_len = int((max_len - len(data)) / 2)
        data = np.concatenate([np.zeros((haft_max_len, size, size, dim)), data], 0)
        data = np.concatenate([data, np.zeros((max_len - len(data), size, size, dim))], 0)
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

def read_image_numpy(image_path, gayscale=False):
    numpy_path = image_path.replace('origin', 'feature')
    numpy_path = numpy_path.replace('.png', '.npy')
    image = np.load(numpy_path)
    image = np.expand_dims(image, 0)
    if not gayscale:
        return image
    else:
        gray = cv2.imread(image_path, 0)
        gray = cv2.resize(gray, (240, 240))
        gray = np.expand_dims(gray, -1)
        gray = np.expand_dims(gray, 0)
        return image, gray

def read_image_numpy_image(image_path, gayscale=False):
    image = cv2.imread(image_path, 0)
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    if not gayscale:
        return image
    else:
        gray = cv2.imread(image_path, 0)
        gray = cv2.resize(gray, (240, 240))
        gray = np.expand_dims(gray, -1)
        gray = np.expand_dims(gray, 0)
        return image, gray

def extract_feature(image):
    feature = feature_extractor(image)
    feature = np.reshape(feature, (128, -1))
    return feature

def extract_feature_512(model,image_stacks, link_stacks, input_folder, output_folder, isOrigin=False):
    features = model.feature_extract(image_stacks)
    write_feature_512(features, link_stacks, input_folder, output_folder, isOrigin)

def generate_image(model, images, links, epoch, folder, isFull=True):
    predictions = None
    if isFull:
        predictions = model(images, training=False)
    else:
        predictions = model.image_generate(images)
    for idx, (image, link) in enumerate(zip(predictions, links)):
        image = np.reshape(image, (32, 640 * 2))
        image = (image * 127.5) + 127.5
        image = image.astype(np.int32)
        link = link.replace(folder, 'image_test')
        file_paths = link.split('/')
        folder_name = '/'.join(file_paths[:-1])
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        cv2.imwrite(f'{folder_name}/image_{epoch}_{idx}.png', image)

def generate_image_only(model, images, epoch, isFull=True):
    predictions = model(images, training=False)
    for idx, image_origin in enumerate(predictions):
        index = int(len(image_origin) / 2)
        image = image_origin[index][:, :, 0]
        image = np.reshape(image, (240, 240))
        image = image * 255
        image = image.astype(np.int32)
        cv2.imwrite(f'images/image_{epoch}_{idx}_FLAIR.png', image)
        image = image_origin[index][:, :, 1]
        image = np.reshape(image, (240, 240))
        image = image * 255
        image = image.astype(np.int32)
        cv2.imwrite(f'images/image_{epoch}_{idx}_T1w.png', image)
        image = image_origin[index][:, :, 2]
        image = np.reshape(image, (240, 240))
        image = image * 255
        image = image.astype(np.int32)
        cv2.imwrite(f'images/image_{epoch}_{idx}_T1wCE.png', image)
        image = image_origin[index][:, :, 3]
        image = np.reshape(image, (240, 240))
        image = image * 255
        image = image.astype(np.int32)
        cv2.imwrite(f'images/image_{epoch}_{idx}_T2w.png', image)

def get_random_sequence(folder_link, isTest=False):
    sequences = np.array([])
    types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    
    isOrder = bool(random.getrandbits(1))
    for type in types:
        sequence = np.array([])
        if os.path.isdir(f"{folder_link}/{type}"):

            folder = os.listdir(f"{folder_link}/{type}")
            
            if isOrder or isTest:
                folder.sort(key=lambda x: int(x.replace('.', '-').split('-')[1]))
            else:
                folder.sort(key=lambda x: int(x.replace('.', '-').split('-')[1]), reverse=True)

            start = len(folder) // 2 - 15
            end = len(folder) // 2 + 15
            if start < 0:
                start = 0
                end = start + 30

            for image in folder[start: end]:
                image = np.load(f"{folder_link}/{type}/{image}")
                image = np.expand_dims(image, 0)
                if len(sequence) == 0:
                    sequence = image
                else:
                    sequence = np.concatenate([sequence, image], 0)
        else:
            sequence = np.array((30, 512))
        
        sequence  = padding_data(sequence, 30)

        if sequences.shape[0] == 0:
            sequences = sequence
        else:
            sequences = np.concatenate([sequences, sequence], 0)
    
    sequences = np.expand_dims(sequences, 0)

    return sequences

def image_generator(data_folder, sample_list, batch_size=128):
    
    images = np.array([])
    grays = np.array([])

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

def image_generator_image(data_folder, sample_list, batch_size=128):
    
    images = np.array([])
    grays = np.array([])

    with tqdm(total=len(sample_list)) as pbar:
        for sample in random_data(sample_list):
            # Reading data (line, record) from the file
            folder_link = f"{data_folder.decode('utf8')}/{sample.decode('utf8')}"
            for sub_folder in ['']:
                for image in os.listdir(f'{folder_link}/{sub_folder}'):
                    try:
                        image, gray = read_image_numpy_image(f'{folder_link}/{sub_folder}/{image}', True)
                        if images.shape[0] >= batch_size:
                            images, grays = random_datas(images, grays)
                            images = images.astype(np.float32)
                            images = images / 255
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
                        print('error ')
                        pass
            pbar.update(1)

def image_generator_image_3d(data_folder, sample_list, batch_size=128):
    
    images = np.array([])
    grays = np.array([])

    with tqdm(total=len(sample_list)) as pbar:
        for sample in random_data(sample_list):
            # Reading data (line, record) from the file
            if sample.decode('utf8') not in ['00109', '00123', '00709']:
                folder_link = f"{data_folder.decode('utf8')}/{sample.decode('utf8')}"
                image_set = None
                image_set_gray = None
                for sub_folder in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
                    image3d = None
                    gray3d = None

                    folder = os.listdir(f'{folder_link}/{sub_folder}')
                    folder.sort(key=lambda x: int(x.replace('_', '-').replace('.', '-').split('-')[-2]))

                    start = len(folder) // 2 - 25
                    end = len(folder) // 2 + 25
                        
                    if start < 0:
                        start = 0
                        end += 50
                    for image in folder[start:end]:
                        image, gray = read_image_numpy_image(f'{folder_link}/{sub_folder}/{image}', True)

                        if image3d is None:
                            image3d = image
                            gray3d = gray
                        else:
                            image3d = np.concatenate([image3d, image], 0)
                            gray3d = np.concatenate([gray3d, gray], 0)
                    
                    image3d = padding_image(image3d, 240, 1, 50)
                    gray3d = padding_image(gray3d, 240, 1, 50)

                    if image_set is None:
                        image_set = image3d
                        image_set_gray = gray3d
                    else:
                        image_set = np.concatenate([image_set, image3d], -1)
                        image_set_gray = np.concatenate([image_set_gray, gray3d], -1)
                        
                image_set = np.expand_dims(image_set, 0)
                image_set_gray = np.expand_dims(image_set_gray, 0)
                                
                if images.shape[0] >= batch_size:
                    images, grays = random_datas(images, grays)
                    images = images.astype(np.float32)
                    images = images / 255
                    grays = grays.astype(np.float32)
                    grays = grays / 255
                    yield images, grays
                    images = image_set
                    grays = image_set_gray
                else:
                    if images.shape[0] == 0:
                        images = image_set
                        grays = image_set_gray
                    else:
                        images = np.concatenate([images, image_set], 0)
                        grays = np.concatenate([grays, image_set_gray], 0)
            pbar.update(1)

def sequence_generator(data_folder, sample_list, labels_list, batch_size=128, isTest=False):

    sequences = np.array([])
    labels = np.array([])

    sample_list, labels_list = random_datas(sample_list, labels_list)

    with tqdm(total=len(sample_list)) as pbar:
        for idx, (image, label) in enumerate(zip(sample_list, labels_list)):
            try:

                folder_link = f"{data_folder.decode('utf8')}/{image.decode('utf8')}"
                label = np.array([int(label)])

                sequence = get_random_sequence(folder_link, isTest)

                if sequences.shape[0] >= batch_size or idx >= len(sample_list) - 1:
                    # if not isTest:
                    #     sequences, labels = random_datas(sequences, labels)
                    yield sequences, labels
                    sequences = sequence
                    labels = label
                else:
                    if sequences.shape[0] == 0:
                        sequences = sequence
                        labels = label
                    else:
                        sequences = np.concatenate([sequences, sequence], 0)
                        labels = np.concatenate([labels, label], 0)
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

def cropped_images(images):
    min=np.array(np.nonzero(images)).min(axis=1)
    max=np.array(np.nonzero(images)).max(axis=1)+1 # --> Thank you @lai321!
    return images[min[0]:max[0],min[1]:max[1],min[2]:max[2], :]

def create_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def read_image(image_path):
    origin_image = cv2.imread(image_path)
    image = np.expand_dims(origin_image, 0)
    return image

def augment_data(folder, feature, n_generated_samples, input_folder, output_folder):
    for type_name in os.listdir(folder + '/' + feature):
        # for idx in range(n_generated_samples):

        aug_input = {}
        image_name = []
        object_type = {}
        file_path_list = {}

        images = None
        file_paths = []

        for id, image in enumerate(os.listdir(folder + '/' + feature + '/' + type_name)):
            # load the image
            file_path = folder + '/' + feature + '/' + type_name + '/' + image
            image = read_image(file_path)
            if images is None:
                images = image
            else:
                images = np.concatenate([images, image], 0)
            file_paths.append(file_path)

        images = cropped_images(images)
        # print('images.shape ', images.shape)
        # for id, (image, file_path) in enumerate(zip(images, file_paths)):
        #     image = create_clahe(image)
        #     if id == 0:
        #         aug_input['image'] = image
        #         file_path_list['image'] = file_path
        #         image_name.append('image')
        #     else:
        #         aug_input['image' + str(id - 1)] = image
        #         file_path_list['image' + str(id - 1)] = file_path
        #         object_type['image' + str(id - 1)] = 'image'
        #         image_name.append('image' + str(id - 1))

        # # aug = strong_aug(object_type)
        # # augmented_data = aug(**aug_input)
        
        # for name in image_name:
        #     image = augmented_data[name]
        #     image = cv2.resize(image, (224, 224))
        #     file_path = file_path_list[name]
        #     file_path = file_path.replace(input_folder, output_folder)
        #     file_paths = file_path.split('/')
        #     file_paths[6] = f'{feature}_{str(idx)}'
        #     file_path = '/'.join(file_paths)
        #     folder_name = '/'.join(file_paths[:-1])
        #     if not os.path.isdir(folder_name):
        #         os.makedirs(folder_name)
        #     if not os.path.isfile(file_path):
        #         cv2.imwrite(file_path, image)
        for image, file_path in zip(images, file_paths):
            # image = create_clahe(image)
            image = cv2.resize(image, (240, 240))
            file_path = file_path.replace(input_folder, output_folder)
            file_paths = file_path.split('/')
            file_paths[6] = feature
            file_path = '/'.join(file_paths)
            folder_name = '/'.join(file_paths[:-1])
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)
            if not os.path.isfile(file_path):
                cv2.imwrite(file_path, image)

def augment_data_split(X_data, y_data, number):
    new_x = []
    new_y = []
    
    for x, y in zip(X_data, y_data):
        if number == 1:
            for idx in range(14):
                if idx == 13:
                    new_x.append(x + '_' + str(idx))
                    new_y.append(y)
        else:
            for idx in range(number):
                new_x.append(x + '_' + str(idx))
                new_y.append(y)

    return new_x, new_y