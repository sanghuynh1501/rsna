import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Reshape, Dense

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

class AutoEncoder(Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    
    self.dense0 = Dense(1024)
    self.dense1 = Dense(512)
    self.dense2 = Dense(16384)
    self.reshape = Reshape((4, 4, 1024))

    self.up60 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.up61 = UpSampling2D(size = (2,2))
    self.conv60 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.up70 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.up71 = UpSampling2D(size = (2,2))
    self.conv70 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.up80 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.up81 = UpSampling2D(size = (2,2))
    self.conv80 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.up90 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.up91 = UpSampling2D(size = (2,2))
    self.conv90 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv92 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv10 = Conv2D(1, 1, activation = 'sigmoid')

  def call(self, x):
    x = self.dense0(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.reshape(x)

    x = self.up60(x)
    x = self.up61(x)
    x = self.conv60(x)
    x = self.conv61(x)

    x = self.up70(x)
    x = self.up71(x)
    x = self.conv70(x)
    x = self.conv71(x)

    x = self.up80(x)
    x = self.up81(x)
    x = self.conv80(x)
    x = self.conv81(x)

    x = self.up90(x)
    x = self.up91(x)
    x = self.conv90(x)
    x = self.conv91(x)
    x = self.conv92(x)
    x = self.conv10(x)

    return x

  def feature_extract(self, x):
    x = self.dense0(x)
    x = self.dense1(x)
    return x

if __name__ == "__main__":
    image = np.ones((32, 100352))
    model = AutoEncoder()
    feature = model.predict(image)
    print('feature.shape ', feature.shape)
    feature = model.feature_extract(image)
    print('feature.shape ', feature.shape)