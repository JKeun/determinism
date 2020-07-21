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
        self.ignore = ignore or []
        self.inputs = sorted(set(self.df.columns) - set(self.target) - set(self.ignore))
        
        self.X = self.df[self.inputs]
        self.y = self.df[self.target]


    def define_problem(self):
        if self.y.dtypes[0] in ['int64', 'float64'] and self.y.nunique()[0] == 2:
            self.problem = "Binary"
        elif self.y.dtypes[0] in ['object', 'bool']:
            self.problem = "Classification"
        else:
            self.problem = "Regression"


    def train_val_split(self, ratio=0.2, random_state=42):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                                                                              test_size=ratio,
                                                                              random_state=random_state)

    
    def data_preprocess(self, X, y, train_set=True):
        if train_set:
            # Data type detection
            numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

            # Data transform
            num_transform = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            cat_transform = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value="Missing")),
                ('oh_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ])

            self.transform_x = ColumnTransformer(transformers=[
                ('num', num_transform, numerical_ix),
                ('cat', cat_transform, categorical_ix)
            ])

            self.trans_X_train = self.transform_x.fit_transform(X)
            
            if self.problem == "Classification":
                self.transform_y = ColumnTransformer(transformers=[
                    ('cat', cat_transform, y.columns)
                ])
                self.trans_y_train = self.transform_y.fit_transform(y)
            else:
                self.trans_y_train = y
        else:
            self.trans_X_val = self.transform_x.transform(X)
            if self.problem == "Classification":
                self.trans_y_val = self.transform_y.transform(y)
            else:
                self.trans_y_val = y
