import pickle
from util import augment_data, augment_data_split, generate_image, generate_image_only
import cv2
import numpy as np
from tf_data import AutoEncoderDataset, AutoEncoderImage3DDataset, AutoEncoderImageDataset
from models import AutoEncoder, AutoEncoderFull, AutoEncoderFull3D, AutoEncoderFull3DNew, AutoEncoderFull3DNew1
import tensorflow as tf

bce_object = tf.keras.losses.MeanSquaredError()
def dice_loss(logits, targets, eps = 1e-9):
    num = targets.shape[0]
    probability = tf.math.sigmoid(logits)
    probability = tf.reshape(probability, (num, -1))
    targets = tf.reshape(targets,(num, -1))
    assert(probability.shape == targets.shape)

    intersection = 2.0 * tf.reduce_sum(probability * targets)
    union = tf.reduce_sum(probability) + tf.reduce_sum(targets)
    dice_score = (intersection + eps) / union

    return 1.0 - dice_score

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryCrossentropy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.BinaryCrossentropy(name='test_accuracy')

model = AutoEncoderFull3DNew1()

DATA_ORIGIN_TRAIN = '/media/sang/Samsung/data_augement_new/train'
DATA_AUGMENT_TRAIN = '/media/sang/Samsung/data_augement_new/train'

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = bce_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
#   train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = bce_object(labels, predictions)

  test_loss(t_loss)
#   test_accuracy(labels, predictions)

EPOCHS = 300

with open('pickle/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
    f.close()

with open('pickle/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    f.close()

with open('pickle/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
    f.close()

with open('pickle/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
    f.close()

# X_train, _ = augment_data_split(X_train, y_train, 14)
# X_test, _ = augment_data_split(X_test, y_test, 14)

batch_size = 4
train_data = AutoEncoderImage3DDataset(DATA_AUGMENT_TRAIN, X_train, batch_size).prefetch(tf.data.AUTOTUNE)
test_data = AutoEncoderImage3DDataset(DATA_AUGMENT_TRAIN, X_test, batch_size).prefetch(tf.data.AUTOTUNE)

checkpoint_path = 'weights/autoencoder_full_3d'

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

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

    for image, gray in train_data:
        train_step(image, gray)

    for image, gray in test_data:
        test_step(image, gray)

    if test_loss.result() < min_loss:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        min_loss = test_loss.result()

    if epoch % 5 == 0:
        for image, _ in test_data:
            generate_image_only(model, image, epoch, isFull=True)
            break

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
    )