import numpy as np

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def feature_extractor(image):
    image = image.astype(np.float32)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation=None,
    )
    feature = resnet.predict(image)
    feature = np.reshape(feature, (feature.shape[0], -1))
    return feature

if __name__ == "__main__":
    image = np.random.uniform(low=0, high=1, size=(128, 224, 224, 3)) * 255
    image = image.astype(np.int32).astype(np.float32)
    feature = feature_extractor(image)
    print('feature ', np.max(feature))