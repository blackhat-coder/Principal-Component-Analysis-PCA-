
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('BLI_21102020083556687.csv')
train.fillna(999, inplace=True)

# train.info()

nums = [x for x in train.columns if train[x].dtype in ['float64', 'int64']]

train_x = train[nums]

train_x = train_x - train_x.mean(axis=0)

class pca():
    def __init__(self, n_components, svd_solver=None):
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, data):
        # data = data.astype('float')
        self.data = data

        self.dim_x , self.dim_y = data.shape
        
        self.u, self.s, self.vt = np.linalg.svd(data)

        self.evr_()

        return self.vt
    
    def transform(self, data):
        try:
            print(f'Compressing Data: ({self.dim_x}, {self.dim_y}) -> ({self.dim_x} , {self.n_components})')
            
            compresed_data = np.dot(data, self.vt.T[::, :self.n_components])
            return compresed_data
        except:
            print('Aplly pca.fit(data) to the data')
        

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)

    def evr_(self):
        self.evr = (self.s / np.sum(self.s)) * 100
        return self.evr

    def evr_plot_(self):

        plt.subplots(figsize=(8, 7))
        sns.set_color_codes(palette='deep')
        sns.set_style('whitegrid')

        sns.barplot(x=[f"pc {i}" for i in range(1, len(self.evr)+1)] , y=self.evr , data=self.data)
        plt.xlabel('principal component')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance Ratio')
        plt.show()
        

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

X_red = pca(n_components=2)
X_red.fit(train_x)

X_red.transform(train_x)