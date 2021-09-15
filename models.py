from i3d_inception import Inception_Inflated3d
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.core import Flatten
from util import create_padding_mask
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling1D, Reshape, Dense, LeakyReLU, Conv2DTranspose, GRU, Conv3DTranspose, ReLU

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

NUM_FRAMES = 40
FRAME_HEIGHT = 240
FRAME_WIDTH = 240
NUM_RGB_CHANNELS = 1
NUM_FLOW_CHANNELS = 2
NUM_CLASSES = 400

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(64, activation='relu')
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inp, training, enc_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        enc_output = self.pooling(enc_output)
        enc_output = self.dense(enc_output)
        class_output = self.classifier(enc_output)

        return class_output

class LSTM_Classifier(tf.keras.Model):
    def __init__(self):
        super(LSTM_Classifier, self).__init__()

        self.reduce = Dense(128, activation='relu')
        # self.lstm0 = LSTM(128, return_sequences=True)
        self.lstm1 = GRU(64, return_sequences=True)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(64, activation='relu')
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inp):
        enc_output = self.reduce(inp)
        # enc_output = self.lstm0(enc_output)  # (batch_size, inp_seq_len, d_model)
        enc_output = self.lstm1(enc_output)
        enc_output = self.pooling(enc_output)
        enc_output = self.dense(enc_output)
        class_output = self.classifier(enc_output)

        return class_output

def CNN_Classifier():
    model = tf.keras.Sequential()
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.leakyRelu = LeakyReLU(alpha=0.2)

        self.dense0 = Dense(2048, activation='relu')
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.dense3 = Dense(256 * 4 * 4)
        self.reshape = Reshape((4, 4, 256))

        self.up0 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.up1 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.up2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.conv0 = Conv2D(1, (3,3), activation='tanh', padding='same')

    def call(self, x):
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.leakyRelu(self.dense3(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.conv0(x)

        return x

    def feature_extract(self, x):
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def image_generate(self, x):
        x = self.leakyRelu(self.dense3(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.conv0(x)
        return x

class AutoEncoderFull(Model):
    def __init__(self):
        super(AutoEncoderFull, self).__init__()
        
        self.leakyRelu = LeakyReLU(alpha=0.2)

        self.resnet = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation=None,
        )
        self.resnet.trainable = False
        self.flatten = Flatten()
        self.dense0 = Dense(2048, activation='relu')
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.dense3 = Dense(256 * 4 * 4)
        self.reshape = Reshape((4, 4, 256))

        self.up0 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.up1 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.up2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.conv0 = Conv2D(1, (3,3), activation='tanh', padding='same')

    def call(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.leakyRelu(self.dense3(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.conv0(x)

        return x

    def feature_extract(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def image_generate(self, x):
        x = self.leakyRelu(self.dense3(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.conv0(x)
        return x

class AutoEncoderFull3D(Model):
    def __init__(self):
        super(AutoEncoderFull3D, self).__init__()
        
        self.leakyRelu = ReLU()

        self.resnet = Inception_Inflated3d(
            include_top=False,
            weights=None,
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
            classes=NUM_CLASSES
        )

        # self.resnet.trainable = False
        self.flatten = Flatten()
        self.dense0 = Dense(2048, activation='relu')
        self.dense1 = Dense(2048, activation='relu')
        self.dense2 = Dense(2048, activation='relu')
        self.dense3 = Dense(1024, activation='relu')
        self.dense4 = Dense(1024, activation='relu')
        self.dense5 = Dense(1024, activation='relu')
        self.dense6 = Dense(3 * 2 * 2 * 128, activation='relu')
        self.reshape = Reshape((3, 2, 2, 128))

        self.up0 = Conv3DTranspose(128, (3,3,3), strides=(1,1,1), padding='valid')
        self.up1 = Conv3DTranspose(128, (4,4,4), strides=(2,2,2), padding='same')
        self.up2 = Conv3DTranspose(128, (4,4,4), strides=(2,2,2), padding='same')
        self.up3 = Conv3DTranspose(1, (4,4,4), strides=(2,2,2), padding='same', activation='sigmoid')

    def call(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.leakyRelu(self.dense6(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.up3(x)

        return x

    def feature_extract(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x
    
    def image_generate(self, x):
        x = self.leakyRelu(self.dense6(x))
        x = self.leakyRelu(self.reshape(x))

        x = self.leakyRelu(self.up0(x))
        x = self.leakyRelu(self.up1(x))
        x = self.leakyRelu(self.up2(x))
        x = self.up3(x)

        return x


class AutoEncoderFull3DNew(Model):
    def __init__(self):
        super(AutoEncoderFull3DNew, self).__init__()
        
        self.conv1 = Conv3D(16, 3, activation='relu')
        self.conv2 = Conv3D(32, 3, activation='relu')
        self.conv3 = Conv3D(96, 2, activation='relu')
        self.pool1 = MaxPool3D(2)
        self.pool2 = MaxPool3D(3)
        self.pool3 = MaxPool3D(2)
        self.enc_linear = Dense(512, activation='relu')
        
        # Decoder
        self.deconv1 = Conv3DTranspose(32, 2, activation='relu')
        self.deconv2 = Conv3DTranspose(16, 3, activation='relu')
        self.deconv3 = Conv3DTranspose(16, 3, activation='relu')
        self.deconv4 = Conv3DTranspose(4, (3, 1, 1), activation='sigmoid')
        self.unpool1 = UpSampling3D(2)
        self.unpool2 = UpSampling3D(3)
        self.unpool3 = UpSampling3D(2)

        self.dec_linear = Dense(103968, activation='relu')

        self.flatten = Flatten()
        self.reshape = Reshape((3, 19, 19, 96))
    
    def encode(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.enc_linear(x)

        return x
    
    def decode(self, x):
        x = self.dec_linear(x)
        x = self.reshape(x)
        
        x = self.unpool1(x)
        x = self.deconv1(x)
        x = self.unpool2(x)
        x = self.deconv2(x)
        x = self.unpool3(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
        
    def call(self, x):
        feature = self.encode(x)
        out = self.decode(feature)
        return out

class AutoEncoderFull3DNew1(Model):
    def __init__(self):
        super(AutoEncoderFull3DNew1, self).__init__()
        
        self.conv1 = Conv3D(16, 3, activation='relu')
        self.conv2 = Conv3D(32, 3, activation='relu')
        self.conv3 = Conv3D(96, 2, activation='relu')
        self.conv4 = Conv3D(128, 2, activation='relu')
        self.pool1 = MaxPool3D(3)
        self.pool2 = MaxPool3D(2)
        self.enc_linear = Dense(512, activation='relu')
        
        # Decoder
        self.deconv1 = Conv3DTranspose(96, 2, activation='relu')
        self.deconv2 = Conv3DTranspose(32, 2, activation='relu')
        self.deconv3 = Conv3DTranspose(32, (1, 2, 2), activation='relu')
        self.deconv4 = Conv3DTranspose(16, 3, activation='relu')
        self.deconv5 = Conv3DTranspose(16, (1, 2, 2), activation='relu')
        self.deconv6 = Conv3DTranspose(4, 3, activation='sigmoid')
        self.unpool1 = UpSampling3D(2)
        self.unpool2 = UpSampling3D(3)

        self.dec_linear = Dense(829440, activation='relu')

        self.flatten = Flatten()
        self.reshape = Reshape((5, 36, 36, 128))
    
    def encode(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.enc_linear(x)

        return x
    
    def decode(self, x):
        x = self.dec_linear(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.unpool1(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.unpool2(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x
        
    def call(self, x):
        feature = self.encode(x)
        out = self.decode(feature)
        return out

if __name__ == "__main__":
    # image = np.ones((32, 100352))
    # model = AutoEncoder()
    # feature = model.predict(image)
    # print('feature.shape ', feature.shape)
    # feature = model.feature_extract(image)
    # print('feature.shape ', feature.shape)

    # num_layers = 2
    # d_model = 64
    # dff = 128
    # num_heads = 4
    # dropout_rate = 0.3

    model = AutoEncoderFull3DNew1()

    image = np.ones((1, 50, 240, 240, 4))
    enc_output = model(image, training=False)
    print('enc_output.shape ', enc_output.shape)

    # temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    # print('temp_input.shape ', temp_input.shape)