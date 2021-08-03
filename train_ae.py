import pickle
from util import generate_image
import cv2
import numpy as np
from tf_data import AutoEncoderDataset
from models import AutoEncoder
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryCrossentropy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.BinaryCrossentropy(name='test_accuracy')

model = AutoEncoder()

DATA_ORIGIN_TRAIN = 'data/origin/train'

@tf.function
def train_step(images, labels):
#   with tf.GradientTape() as tape:
#     # training=True is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(images, training=True)
#     loss = loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  predictions = model(images, training=False)
  loss = loss_object(labels, predictions)

  train_loss(loss)
#   train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
#   test_accuracy(labels, predictions)

EPOCHS = 100

with open('pickle/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
    f.close()

with open('pickle/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    f.close()

batch_size = 32
train_data = AutoEncoderDataset(DATA_ORIGIN_TRAIN, X_train, batch_size).prefetch(tf.data.AUTOTUNE)
test_data = AutoEncoderDataset(DATA_ORIGIN_TRAIN, X_test, batch_size).prefetch(tf.data.AUTOTUNE)

checkpoint_path = 'weights/autoencoder_v2'

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

min_loss = float('inf')

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    # train_accuracy.reset_states()
    test_loss.reset_states()
    # test_accuracy.reset_states()

    # for image, gray in train_data:
    #     train_step(image, gray)

    # for image, gray in test_data:
    #     test_step(image, gray)

    # if test_loss.result() < min_loss:
    #     ckpt_save_path = ckpt_manager.save()
    #     print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                         ckpt_save_path))
    #     min_loss = test_loss.result()

    if epoch % 5 == 0:
        for image, _ in test_data:
            generate_image(model, image, epoch, isFull=True)
            break

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
    )