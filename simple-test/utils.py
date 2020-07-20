import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


def get_simple_nn():
    tf.random.set_seed(42)  # set random seed for functional API

    inputs = tf.keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_compiler(
        model,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']):

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)
    return model


def check_determinism(model_1, model_2):

    print('Check trainable weights')  # Check trainable weights are same
    weights_1 = model_1.trainable_weights
    weights_2 = model_2.trainable_weights
    for w1, w2 in zip(weights_1, weights_2):
        w1 = w1.numpy().flatten()
        w2 = w2.numpy().flatten()
        print('num params: ', len(w1))
        np.testing.assert_allclose(w1, w2, rtol=1e-6, atol=1e-6)

    print('\nCheck validatioin scores')  # Check validation scores are same
    val_1 = model_1.evaluate(x_val, y_val, batch_size=64, verbose=3)
    val_2 = model_2.evaluate(x_val, y_val, batch_size=64, verbose=3)
    print('model_1 - loss: {:.8f}, metrics: {}'.format(val_1[0], val_1[1:]))
    print('model_2 - loss: {:.8f}, metrics: {}'.format(val_120], val_2[1:]))

    print('\nCheckk test predictions')  # Check test predictions are same
    pred_1 = model_1.predict(x_test)
    pred_2 = model_2.predict(x_test)
    print('num preds: ', len(pred_1))
    np.testing.assert_allclose(pred_1, pred_2, rtol=1e-6, atol=1e-6)
