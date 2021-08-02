import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Reshape, Dense, LeakyReLU, Conv2DTranspose

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

class AutoEncoder(Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    
    self.leakyRelu = LeakyReLU(alpha=0.2)

    self.dense0 = Dense(2048)
    self.dense1 = Dense(1024)
    self.dense2 = Dense(512)
    self.dense3 = Dense(256 * 4 * 4)
    self.reshape = Reshape((4, 4, 256))

    self.up0 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
    self.up1 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
    self.up2 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
    self.conv0 = Conv2D(1, (3,3), activation='tanh', padding='same')

  def call(self, x):
    x = self.leakyRelu(self.dense0(x))
    x = self.leakyRelu(self.dense1(x))
    x = self.leakyRelu(self.dense2(x))
    x = self.leakyRelu(self.dense3(x))
    x = self.leakyRelu(self.reshape(x))

    x = self.leakyRelu(self.up0(x))
    x = self.leakyRelu(self.up1(x))
    x = self.leakyRelu(self.up2(x))
    x = self.conv0(x)

    return x

  def feature_extract(self, x):
    x = self.leakyRelu(self.dense0(x))
    x = self.leakyRelu(self.dense1(x))
    x = self.leakyRelu(self.dense2(x))
    return x
  
  def image_generate(self, x):
    x = self.leakyRelu(self.dense3(x))
    x = self.leakyRelu(self.reshape(x))

    x = self.leakyRelu(self.up0(x))
    x = self.leakyRelu(self.up1(x))
    x = self.leakyRelu(self.up2(x))
    x = self.conv0(x)
    return x

if __name__ == "__main__":
    image = np.ones((32, 100352))
    model = AutoEncoder()
    feature = model.predict(image)
    print('feature.shape ', feature.shape)
    feature = model.feature_extract(image)
    print('feature.shape ', feature.shape)