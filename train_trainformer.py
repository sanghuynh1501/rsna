import pickle
from util import augment_data, generate_image, sequence_generator
from tf_data import TransformerDataset
from models import Transformer
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

DATA_FEATURE_TRAIN = 'data/feature_512/train'
DATA_FEATURE_TEST = 'data/test_origin/test'

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

num_layers = 2
d_model = 64
dff = 128
num_heads = 4
dropout_rate = 0.3

model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    pe_input=1000,
    rate=dropout_rate
)

X_train, y_train = augment_data(X_train, y_train)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_acc = tf.keras.metrics.BinaryCrossentropy(name='train_acc')
train_auc = tf.keras.metrics.AUC()

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_acc = tf.keras.metrics.BinaryCrossentropy(name='test_acc')
test_auc = tf.keras.metrics.AUC()

batch_size = 32
train_data = TransformerDataset(DATA_FEATURE_TRAIN, X_train, y_train, 'FLAIR', batch_size, False).prefetch(tf.data.AUTOTUNE)
test_data = TransformerDataset(DATA_FEATURE_TEST, X_test, y_test, 'FLAIR', batch_size, True).prefetch(tf.data.AUTOTUNE)

def train_step(images, masks, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, True, masks)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_auc(labels, predictions)

def test_step(images, masks, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, False, masks)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_auc(labels, predictions)

checkpoint_path = 'weights/transformer'

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

min_loss = float('inf')

EPOCHS = 100

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_auc.reset_states()
    test_loss.reset_states()
    test_auc.reset_states()

    for image, mask, label in train_data:
        train_step(image, mask, label)

    for image, mask, labe in test_data:
        test_step(image, mask, labe)

    if test_loss.result() < min_loss:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        min_loss = test_loss.result()

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Train Acc: {train_auc.result()}, '
        f'Test Acc: {test_auc.result()}, '
    )