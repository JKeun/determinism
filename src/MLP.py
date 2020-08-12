import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_single_mlp(X, y, problem, hidden_layer=1, unit=16):
    tf.random.set_seed(42)
    
    tf.keras.backend.clear_session()  # clear graph session
    model = keras.Sequential()
    # input layer
    model.add(keras.layers.Input(shape=(X.shape[1],)))
    # hidden layer
    for _ in range(hidden_layer):
        model.add(keras.layers.Dense(unit, activation='relu'))
    # output layer
    if problem == "Regression":
        model.add(keras.layers.Dense(1))
    elif problem == "Binary":
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(keras.layers.Dense(y.shape[1], activation='softmax'))
    
    return model


def compile_model(problem, model, optimizer='adam', lr=0.01):
    # automatically set loss and metrics according to problem
    if problem == "Regression":
        loss = keras.losses.MSE
        metrics = ['MSE', 'MAE']
    elif problem == "Binary":
        loss = keras.losses.binary_crossentropy
        metrics = ['accuracy']
    else:
        loss = keras.losses.categorical_crossentropy
        metrics = ['accuracy']
    
    # match optimizer argument to optimizer class
    optimizer_classes = {'adadelta': keras.optimizers.Adadelta, 'sgd': keras.optimizers.SGD,
                         'adam': keras.optimizers.Adam, 'adagrad': keras.optimizers.Adagrad,
                         'adamax': keras.optimizers.Adamax, 'rmsprop': keras.optimizers.RMSprop}
    optimizer_class = optimizer_classes[optimizer]
    
    optimizer_info = {'optimizer': optimizer,
                      'lr': lr}
    
    opt = optimizer_class(learning_rate=lr)
    model.compile(optimizer=opt,
                           loss=loss,
                           metrics=metrics)


def train_model(model, X_train, y_train, X_val, y_val,
                batch_size=None, epochs=1, verbose=0, callbacks=None,
                shuffle=True, steps_per_epoch=None):
    # set callbacks; EarlyStopping
    if callbacks:
        callbacks = callbacks
    else:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5,
                                                   restore_best_weights=False)]
    
    model.fit(x=X_train, y=y_train,
              batch_size=batch_size, epochs=epochs,
              verbose=verbose, callbacks=callbacks,
              validation_data=(X_val, y_val), shuffle=shuffle)
    
    val_loss = model.evaluate(X_val, y_val, verbose=verbose)
    print("{} model is trained. best val loss is: {}".format(model.name, val_loss))
    
    return model, val_loss


def auto_fit(problem, X_train, y_train, X_val, y_val,
             hidden_layers=[1], units=[16],
             optimizers=['adam'], lrs=[0.001],
             batch_size=None, epochs=1, verbose=0,
             callbacks=None, shuffle=True,
             steps_per_epoch=None,
             use_all=False):

    if use_all:
        hidden_layers = [1, 2, 3]
        units = [16, 32, 64, 128, 256]
        optimizers = ['adam', 'adadelta', 'adamax', 'adagrad', 'sgd', 'rmsprop']
        lrs = [0.001, 0.01, 0.02, 0.1]
    else:
        hidden_layers = hidden_layers
        units = units
        optimizers = optimizers
        lrs = lrs
        
    models = []
    val_losses = []
    param_info = []
    param_grid = [hidden_layers, units, optimizers, lrs]
    for param_tuple in itertools.product(*param_grid):
        hidden_layer = param_tuple[0]
        unit = param_tuple[1]
        optimizer = param_tuple[2]
        lr = param_tuple[3]
        
        param_dict = {'hidden_layer': hidden_layer,
                      'unit': unit,
                      'optimizer': optimizer,
                      'lr': lr}
        
        model = get_single_mlp(X_train, y_train, problem,
                              hidden_layer=hidden_layer, unit=unit)
        
        compile_model(problem, model, optimizer=optimizer, lr=lr)
        
        model, val_loss = train_model(model, X_train, y_train, X_val, y_val,
                                      batch_size=batch_size, epochs=epochs, verbose=verbose,
                                      callbacks=callbacks, shuffle=shuffle,
                                      steps_per_epoch=steps_per_epoch)
    
        models.append(model)
        val_losses.append(val_loss[0])
        param_info.append(param_dict)
        
    return models, param_info, val_losses


def select_best_model(trained_models, val_losses, models_info):
    best_idx = np.nanargmin(val_losses)
    best_model = trained_models[best_idx]
    best_model_info = models_info[best_idx]
    return best_model, best_model_info
