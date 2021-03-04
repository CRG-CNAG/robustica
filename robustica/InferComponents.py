#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
# 

from robustica.examples import sampledata
from sklearn.decomposition import PCA
import numpy as np

class InferComponents():
    """
    Estimate the number of principal components needed to explain a certain 
    amount of variance.
    """
    def __init__(self, max_variance_explained_ratio=0.99, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None):
        # PCA parameters
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        
        # inference
        self.max_variance_explained_ratio=max_variance_explained_ratio
        
        # initialize PCA
        self.pca = PCA(
            n_components=self.n_components, 
            copy=self.copy, 
            whiten=self.whiten, 
            svd_solver=self.svd_solver, 
            tol=self.tol, 
            iterated_power=self.iterated_power, 
            random_state=self.random_state
        )
        
        
    def fit(self, X):
        self.pca.fit(X)
        self.cumsum_ = np.cumsum(self.pca.explained_variance_ratio_)
        self.inferred_components_ = np.min(np.where(self.cumsum_ >= self.max_variance_explained_ratio)[0])
        
        
    def predict(self, X):
        return self.inferred_components_
        
        
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)