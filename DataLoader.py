import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


def get_processed_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.
    
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    
    return x_train, y_train, x_val, y_val, x_test, y_test
