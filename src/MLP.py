import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class MLP:
    
    def __init__(self, X, y, problem="Regression"):
        tf.random.set_seed(42)
        self.problem = problem
        self.X = X
        self.y = y


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
            model.add(keras.layers.Dense(16, input_shape=(self.X.shape[1],)))

            # hidden layer block
            for _ in range(structure_param['hidden_layers']):
                model.add(keras.layers.Dense(structure_param['units'], activation='relu'))

            # output layer
            if self.problem == 'Regression':
                model.add(keras.layers.Dense(1))
            elif self.problem == 'Binary':
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(self.y.shape[1], activation='softmax'))

            self.structures.append(model)
            self.structures_info.append(structure_param)

            
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

    
    def compile_model(self):
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

    
    def train_models(self, models, X_train, y_train, X_val=None, y_val=None,
                     batch_size=None, epochs=1, verbose=0, callbacks=None,
                     shuffle=True, steps_per_epoch=None):
        
        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=5, restore_best_weights=True)

        self.trained_models = []
        self.val_losses = []
        for model in models:
            model.fit(x=X_train, y=y_train,
                      batch_size=batch_size, epochs=epochs,
                      verbose=verbose, callbacks=callbacks,
                      validation_data=(X_val, y_val), shuffle=shuffle)
            
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            self.trained_models.append(model)
            self.val_losses.append(val_loss[0])
            print("{} model is trained. best val loss is : {}".format(model.name, val_loss))
                
    
def select_best_model(trained_models, val_losses, models_info):
    best_idx = np.nanargmin(val_losses)
    best_model = trained_models[best_idx]
    best_model_info = models_info[best_idx]
    return best_model, best_model_info
