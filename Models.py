import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


def get_simple_nn():
    inputs = tf.keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_simple_nn(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['sparse_categorical_accuracy']):
    model = get_simple_nn()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model
