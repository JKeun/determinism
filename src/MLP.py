import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class MLP:

    def __init__(self, problem="Regression"):
        self.problem = problem
        tf.random.set_seed(42)

    def build_structure(self, max_hidden_layers=1, units=[16], use_all=False):
        self.structures = []
        self.structures_info = []
        if use_all:
            self.max_hidden_layers = 3
            self.units = [16, 32, 64, 128, 256]
        else:
            self.max_hidden_layers = max_hidden_layers
            self.units = units

        grid = [np.arange(self.max_hidden_layers)+1, self.units]
        for param_tuple in itertools.product(*grid):
            structure_param = {'hidden_layers': param_tuple[0],
                                'units': param_tuple[1]}

            # input layer
            model = keras.Sequential()
            model.add(keras.layers.Dense(16, input_shape=(trans_X.shape[1],)))

            # hidden layer block
            for _ in range(structure_param['hidden_layers']):
                model.add(keras.layers.Dense(structure_param['units'], activation='relu'))

            # output layer
            if problem == 'Regression':
                model.add(keras.layers.Dense(1))
            elif problem == 'Binary':
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(trans_y.shape[1], activation='softmax'))

            self.structures.append(model)
            self.structures_info.append(structure_param)

        return self.structures, self.structures_info

    def create_optimizer(self, optimizers=['adam'], lrs=[0.01], use_all=False):
        self.created_optimizers = []
        self.optimizers_info = []
        self.optimizers = optimizers
        self.lrs = lrs

        self.optimizer_classes = {'adadelta': keras.optimizers.Adadelta, 'sgd': keras.optimizers.SGD,
                                  'adam': keras.optimizers.Adam, 'adagrad': keras.optimizers.Adagrad,
                                  'adamax': keras.optimizers.Adamax, 'rmsprop': keras.optimizers.RMSprop}

        if use_all:
            self.lrs = [0.001, 0.01, 0.02, 0.1]
            opt_grid = [self.optimizer_classes.keys(), self.lrs]
        else:
            opt_grid = [self.optimizers, self.lrs]

        for opt_tuple in itertools.product(*opt_grid):
            opt_param = {
                'optimizer_name': opt_tuple[0],
                'lr': opt_tuple[1]
            }

            opt_class = self.optimizer_classes.get(opt_param['optimizer_name'])
            self.created_optimizers.append(opt_class(opt_param['lr']))
            self.optimizers_info.append(opt_param)

        return self.created_optimizers, self.optimizers_info

    def _compile_model(self):
        if self.problem == "Regression":
            self.loss = keras.losses.MSE
            self.metrics = ['MSE', 'MAE']
        elif self.problem == "Binary":
            self.loss = keras.losses.binary_crossentropy
            self.metrics = ['accuracy']
        else:
            self.loss = keras.losses.categorical_crossentropy
            self.metrics = ['accuracy']

        self.compiled_models = []
        self.compiled_models_info = []

        compile_grid = [zip(self.structures, self.structures_info), zip(self.created_optimizers ,self.optimizers_info)]
        for compile_tuple in itertools.product(*compile_grid):
            compile_param = {'model': compile_tuple[0][0],
                             'optimizer': compile_tuple[1][0]}
            model_info = {'structure_info': compile_tuple[0][1],
                           'optimizer_info': compile_tuple[1][1]}

            model_body = compile_param['model']
            model = keras.models.clone_model(model_body)
            model.compile(optimizer=compile_param['optimizer'],
                               loss=self.loss,
                               metrics=self.metrics)

            self.compiled_models.append(model)
            self.compiled_models_info.append(model_info)

        return self.compiled_models, self.compiled_models_info

    def train_models(self, models, X_train, y_train, X_val=None, y_val=None,
                     batch_size=None, epochs=1, verbose=0, callbacks=None,
                     shuffle=True, steps_per_epoch=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.trained_models = []
        self.val_losses = []
        for model in models:
            model.fit(x=self.X_train, y=self.y_train,
                      batch_size=batch_size, epochs=epochs,
                      verbose=verbose, callbacks=callbacks,
                      validation_data=(self.X_val, self.y_val), shuffle=shuffle)

            val_loss = model.evaluate(self.X_val, self.y_val, verbose=0)
            self.trained_models.append(model)
            self.val_losses.append(val_loss[0])
            print("{} model is trained. best val loss is : {}".format(model.name, val_loss))

        return self.trained_models


def select_best_model(trained_models, val_losses, models_info):
    best_idx = np.argmin(val_losses)
    best_model = trained_models[best_idx]
    best_model_info = compiled_models_info[best_idx]
    return best_model, best_model_info
