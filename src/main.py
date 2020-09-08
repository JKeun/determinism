import sys
import time
import numpy as np

from DataSource import DataSource
from MLP import get_single_mlp, compile_model, train_model, auto_fit, select_best_model


# Datasource settings
filename = "../data/Titanic_train.csv"
target = 'Survived'
ignore_fields = ['Name', 'Cabin', 'Ticket']

# Network settings
use_all = False
hidden_layers = [1, 2]
units = [16, 32]
optimizers = ['adam']
lrs = [0.01]


ds = DataSource(filename)
ds.data_load_split(target=[target],
                   ignore=ignore_fields)
ds.define_problem()
ds.train_val_split(ratio=0.2, random_state=42)

ds.data_preprocess(ds.X_train, ds.y_train, train_set=True)
ds.data_preprocess(ds.X_val, ds.y_val, train_set=False)

X_train, y_train = ds.trans_X_train, ds.trans_y_train
X_val, y_val = ds.trans_X_val, ds.trans_y_val


def train(hidden_layers, units, optimizers, lrs, use_all=False):
    # fit models
    start_time = time.time()
    models, param_info, val_losses = auto_fit(ds.problem,
                                              X_train, y_train,
                                              X_val, y_val,
                                              hidden_layers=hidden_layers, units=units,
                                              optimizers=optimizers, lrs=lrs,
                                              batch_size=64, epochs=10,
                                              verbose=0,
                                              use_all=use_all)
    print("===== %s mins =====" % ((time.time() - start_time) / 60))

    best_model, best_model_info = select_best_model(models, val_losses, param_info)
    print("best_model_info: ", best_model_info)


def check(hidden_layers, units, optimizers, lrs, use_all=False):
    models_list = []
    val_losses_list = []
    best_model_list = []
    for i in range(2):
        start_time = time.time()
        models, param_info, val_losses = auto_fit(ds.problem,
                                                  X_train, y_train,
                                                  X_val, y_val,
                                                  hidden_layers=hidden_layers, units=units,
                                                  optimizers=optimizers, lrs=lrs,
                                                  batch_size=64, epochs=10,
                                                  verbose=0,
                                                  use_all=use_all)
        print("#%s train is over ===== %.4f mins =====\n" % (i+1, ((time.time() - start_time) / 60)))
        best_model, best_model_info = select_best_model(models, val_losses, param_info)

        models_list.append(models)
        val_losses_list.append(val_losses)
        best_model_list.append(best_model_info)

    print("Two val losses (%s length) are same? : " % (len(val_losses_list[0])), np.array_equal(val_losses_list[0], val_losses_list[1]))
    print("1st best_model_info: ", best_model_list[0])
    print("2nd best_model_info: ", best_model_list[1])



if __name__ == '__main__':
    if sys.argv[1] == '--train':
        train(hidden_layers, units, optimizers, lrs, use_all)
    elif sys.argv[1] == '--check':
        check(hidden_layers, units, optimizers, lrs, use_all)
