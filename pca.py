
import numpy as np
import pandas as pd

train = pd.read_csv('BLI_21102020083556687.csv')
train.fillna(999, inplace=True)

# train.info()

nums = [x for x in train.columns if train[x].dtype in ['float64', 'int64']]

train_x = train[nums]

train_x = train_x - train_x.mean(axis=0)

u, s , vt = np.linalg.svd(train_x)

pca_1 = vt.T[:, 1]
pca_2 = vt.T[:, 2]

print(pca_1)
print(" ")

print(pca_2)


class pca():

    def __init__(self, n_components, svd_solver=None):
        self.n_compnents = n_components
        self.svd_solver = svd_solver

    def fit(self, data):
        pass
    
    def transform(self, data):
        pass

    def fit_transform(self, data):
        pass
    
    def evr_(self):
        pass

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass