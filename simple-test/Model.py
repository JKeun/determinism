import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


class MLP(tf.keras.Model):

    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(32, activation='relu', name='dense_2')
        self.classifier = layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.classifier(x)
