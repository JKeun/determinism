from src.DataSource import DataSource
from src.MLP import get_single_mlp, compile_model, train_model, auto_fit, select_best_model

filename = "./data/titanic_train.scv"
ds = DataSource(filename)
ds.data_load_split(target=['Survived'],
                   ignore=['Name', 'Cabin', 'Ticket'])
ds.define_problem()
ds.train_val_split(ratio=0.2, random_state=42)

ds.data_preprocess(ds.X_train, ds.y_train, train_set=True)
ds.data_preprocess(ds.X_val, ds.y_val, train_set=False)

X_train = ds.trans_X_train
y_train = ds.trans_y_train
X_val = ds.trans_X_val
y_val = ds.trans_y_val

import time
start_time = time.time()
models, param_info, val_losses = auto_fit(ds.problem, X_train, y_train, X_val, y_val,
                                          hidden_layers=[1], units=[16],
                                          optimizers=['adam'], lrs=[0.01],
                                          batch_size=64, epochs=10, verbose=0,
                                          use_all=True)
print("==== %s mins ====" % ((time.time() - start_time) / 60))

best_model, best_model_info = select_best_model(models, val_losses, param_info)
print("\n", best_model_info)
