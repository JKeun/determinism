import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataSource:

    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def data_load_split(self, target=None, ignore=None):
        self.target = target
        self.ignore = ignore
        self.inputs = sorted(set(self.df.columns) - set(self.target) - set(self.ignore))

        self.X = self.df[self.inputs]
        self.y = self.df[self.target]

        return self.X, self.y

    def define_problem(self):
        if self.y.dtypes[0] in ['int64', 'float64'] and self.y.nunique()[0] == 2:
            self.problem = "Binary"
        elif self.y.dtypes[0] in ['object', 'bool']:
            self.problem = "Classification"
        else:
            self.problem = "Regression"

        return self.problem

    def data_preprocess(self, X, y, problem="Regression"):

        # Data type detection
        numerical_ix = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_ix = self.X.select_dtypes(include=['object', 'bool']).columns

        # Data transform
        num_transform = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        cat_transform = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value="Missing")),
            ('oh_encoder', OneHotEncoder(sparse=False))
        ])

        transform_x = ColumnTransformer(transformers=[
            ('num', num_transform, numerical_ix),
            ('cat', cat_transform, categorical_ix)
        ])

        if problem == "Regression" or "Binary":
            transform_y = ColumnTransformer(transformers=[
                ('num', Normalizer(), y.columns)
            ])
        else:
            transform_y = ColumnTransformer(transformers=[
                ('cat', cat_transform, y.columns)
            ])

        trans_X = transform_x.fit_transform(self.X)
        trans_y = transform_y.fit_transform(self.y)

        return trans_X, trans_y

    def train_val_split(self, X, y, ratio=0.2, random_state=42):
        return train_test_split(X, y, test_size=ratio, random_state=random_state)
