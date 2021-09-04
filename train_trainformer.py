import pickle
from util import augment_data, augment_data_split, generate_image, sequence_generator
from tf_data import TransformerDataset
from models import CNN_Classifier
import tensorflow as tf

DATA_FEATURE_TRAIN = 'data/feature_512/train'
DATA_FEATURE_TEST = 'data/feature_512/train'

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

model = CNN_Classifier()

# X_train, y_train = augment_data_split(X_train, y_train)
# X_test, y_test = augment_data_split(X_test, y_test)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_acc = tf.keras.metrics.BinaryCrossentropy(name='train_acc')
train_auc = tf.keras.metrics.AUC()

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_acc = tf.keras.metrics.BinaryCrossentropy(name='test_acc')
test_auc = tf.keras.metrics.AUC()

batch_size = 32
train_data = TransformerDataset(DATA_FEATURE_TRAIN, X_train, y_train, batch_size, False).prefetch(tf.data.AUTOTUNE)
test_data = TransformerDataset(DATA_FEATURE_TEST, X_test, y_test, batch_size, True).prefetch(tf.data.AUTOTUNE)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_auc(labels, predictions)

def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_auc(labels, predictions)

checkpoint_path = 'weights/lstm'

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

    for image, label in train_data:
        train_step(image, label)

    for image, labe in test_data:
        test_step(image, labe)

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