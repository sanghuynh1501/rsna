import os
import csv
import numpy as np
import matplotlib.pyplot as plt

DATA_FEATURE = 'data/feature/train'

positive_count = 0
negative_count = 0

def plot_bar(folder_name='FLAIR'):
    print(f'{folder_name}:')
    data = []
    total_zero = 0
    max_length = 0
    for item in os.listdir(DATA_FEATURE):
        feature_length =  0
        if os.path.isdir(f'{DATA_FEATURE}/{item}/{folder_name}'):
            feature_length = len(os.listdir(f'{DATA_FEATURE}/{item}/{folder_name}'))
            if feature_length > max_length:
                max_length = feature_length
        else:
            total_zero += 1
        data.append(feature_length)
    y_pos = np.arange(len(data))
    print('total_zero ', total_zero)
    print('max_length ', max_length)
    plt.bar(y_pos, data)
    plt.show()

with open('data/train_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if int(row[1]) == 1:
                positive_count += 1
            else:
                negative_count += 1
            line_count += 1
    print(f'Processed {line_count} lines.')

print('positive_count ', positive_count)
print('negative_count ', negative_count)

plot_bar(folder_name='FLAIR')
plot_bar(folder_name='T1w')
plot_bar(folder_name='T1wCE')
plot_bar(folder_name='T2w')