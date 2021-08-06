import os
from util import augment_data
from tqdm import tqdm

STACK_SIZE = 256
DATA_ORIGIN = 'data/origin'
DATA_AUGEMENT = 'data/augement'

data = os.listdir(DATA_ORIGIN)
test_length = len(os.listdir(DATA_ORIGIN + '/' + 'test'))
train_length = len(os.listdir(DATA_ORIGIN + '/' + 'train'))

with tqdm(total=(test_length + train_length)) as pbar:
    for train_type in data:
        for item in os.listdir(DATA_ORIGIN + '/' + train_type):
            augment_data(DATA_ORIGIN + '/' + train_type, item, DATA_ORIGIN, DATA_AUGEMENT,  10)
            pbar.update(1)