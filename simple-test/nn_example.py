import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from tensorflow import keras


df = pd.read_csv("../../data/titanic/titanic_train.csv")

target = "Survived"
ignore = ["Name", "Cabin", "Ticket"]
inputs = sorted(set(df.columns) - set([target]) - set(ignore))


# Data type detection
numerical_ix = df[inputs].select_dtypes(include=['int64', 'float64']).columns
categorical_ix = df[inputs].select_dtypes(include=['object', 'bool']).columns


# Data transforms
cat_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
        ('oh_encoder', OneHotEncoder(sparse=False))
    ])
num_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])

transform_x = ColumnTransformer(transformers=[
        ('cat', cat_transform, categorical_ix),
        ('num', num_transform, numerical_ix)
    ])
transform_y = ColumnTransformer(transformers=[
        ('num', Normalizer(), pd.Index([target]))
    ])

trans_x = transform_x.fit_transform(df[inputs])
trans_y = transform_y.fit_transform(df)

x_train, x_val, y_train, y_val = train_test_split(trans_x, trans_y, test_size=0.2, random_state=42)


# Neural Network
#tf.random.set_seed(42)
nn_inputs = keras.Input(shape=(x_train.shape[1],), name='input')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(nn_inputs)
x = keras.layers.Dense(32, activation='relu', name='dense_2')(x)
nn_outputs = keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)
model = keras.Model(inputs=nn_inputs, outputs=nn_outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)
print(val_acc)
