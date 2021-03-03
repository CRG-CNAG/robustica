#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
# 

from examples import sampledata
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.cluster import DBSCAN
import multiprocessing as mp
from scipy import stats, sparse


class Sastry():
    """
    special distance matrix with precomputed DBSCAN.
    """
    def __init__(self, **kws):
        self.clustering = DBSCAN(metric='precomputed', **kws)
    
    def fit(self, X):
        # compute similarity matrix
        dist = abs(np.dot(X,X.T)) # the input will be transposed
        dist[dist < .5] = 0
        D = 1 - np.round(dist, 13) # floating point imprecision
        
        # cluster
        self.clustering.fit(D)
        self.labels_ = self.clustering.labels_
    
    
def Icasso():
    pass

        
class RobustICA():
    """
    Class to perform robust Independent Component Analysis (ICA) using different
    methods to cluster together the independent components computed via 
    `sklearn.decomposition.FastICA`.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If None is passed, all are used.
        
    algorithm : {'parallel', 'deflation'}, default='parallel'
        Apply parallel or deflational algorithm for FastICA.
    
    whiten : bool, default=True
        If whiten is false, the data is already considered to be
        whitened, and no whitening is performed.
    
    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example::
            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)
    
    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.
    
    max_iter : int, default=200
        Maximum number of iterations during fit.
    
    tol : float, default=1e-4
        Tolerance on update at each iteration.
    
    w_init : ndarray of shape (n_components, n_components), default=None
        The mixing matrix to be used to initialize the algorithm.
    
    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.
    
    method : {'fcluster', 'DBSCAN'}, default=None
        The clustering method to use in order to build the robust components.
        'fcluster' corresponds to the Icasso approach described in .
        'DBSCAN' corresponds to the approach described in 
    
    method_kws : dict, default={}
        Dictionary of parameters for the clustering method function.
        
        
    Attributes
    ----------
    
    
    Examples
    --------
    from robustica import RobustICA, sampledata
    
    X = sampledata
    rica = RobustICA()
    S = rica.fit_transform(X)
    A = rica.mixing_
    
    Notes
    -----
    Icasso procedure based on
    *Himberg, Johan, Aapo Hyvärinen, and Fabrizio Esposito. "Validating the 
    independent components of neuroimaging time series via clustering and 
    visualization." Neuroimage 22.3 (2004): 1214-1222.*
    
    DBSCAN procedure based on
    *Sastry, Anand V., et al. "The Escherichia coli transcriptome mostly 
    consists of independently regulated modules." 
    Nature communications 10.1 (2019): 1-14.*
    
    """
    def __init__(self, n_components=None, algorithm='parallel', whiten=True, 
                 fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, 
                 w_init=None, random_state=None, n_jobs=None,
                 robust_iter=100, robust_method='DBSCAN', robust_kws={}):
        
        # parameters for FastICA
        if max_iter < 1:
            raise ValueError("max_iter should be greater than 1, got "
                             "(max_iter={})".format(max_iter))
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
        
        # initialize FastICA
        self.ica = FastICA(
             n_components=self.n_components, algorithm=self.algorithm, 
             whiten=self.whiten, fun=self.fun, fun_args=self.fun_args, 
             max_iter=self.max_iter, tol=self.tol, 
             w_init=self.w_init, random_state=self.random_state
        )
        
        # parameters for robust procedure
        self.n_jobs = n_jobs
        self.robust_iter = robust_iter
        self.robust_method = robust_method
        self.robust_kws = robust_kws
        self.cluster_funcs = {
            'sastry': Sastry,
            'icasso': Icasso,
            'DBSCAN': DBSCAN
        }
        
        # initialize clustering function
        self._prep_cluster_func()
        self.clustering = self.cluster_func(**self.robust_kws)
        
    
    def _run_ica(self, X):
        S = self.ica.fit_transform(X)
        A = self.ica.mixing_
        return {'S':S, 'A':A}
    

    def _iterate_ica(self, X):
        """
        Example
        -------
        X = sampledata
        rica = RobustICA(n_jobs=10, n_components=10)
        rica._iterate_ica(X.values)
        rica.S_all.shape
        rica.A_all.shape
        """
        # iterate
        pool = mp.pool.ThreadPool(self.n_jobs)
        args = [X for it in range(self.robust_iter)]
        result = pool.map(self._run_ica, args)
        pool.close()
        pool.join()
        
        # prepare output
        self.S_all = np.hstack([r['S'] for r in result])
        self.A_all = np.hstack([r['A'] for r in result])
        
        
    def _prep_cluster_func(self):
        if isinstance(self.robust_method, str):
            self.cluster_func = self.cluster_funcs[self.robust_method]
        else:
            self.cluster_func = self.robust_method
            
        
    def _compute_centroids(self):
        """
        Taken from https://github.com/SBRG/precise-db/blob/master/scripts/cluster_components.py
        """
        # put clusters together and correct components direction
        S = []
        A = []
        sumstats = []
        labels = np.array(self.clustering.labels_)
        for label in np.unique(labels):
            # subset
            idx = labels == label
            S_clust = self.S_all[:,idx]
            A_clust = self.A_all[:,idx]
            
            # first item is base component
            Svec0 = S_clust[:,0]
            Avec0 = A_clust[:,0]

            # Make sure base component is facing positive
            if abs(min(Svec0)) > max(Svec0):
                Svec0 = -Svec0
                Avec0 = -Avec0

            S_single = [Svec0]
            A_single = [Avec0]
            # Add in rest of components
            for j in range(1,S_clust.shape[1]):
                Svec = S_clust[:,j]
                Avec = A_clust[:,j]
                if stats.pearsonr(Svec,Svec0)[0] > 0:
                    S_single.append(Svec)
                    A_single.append(Avec)
                else:
                    S_single.append(-Svec)
                    A_single.append(-Avec)

            # save centroids
            S.append(np.array(S_single).T.mean(axis=1))
            A.append(np.array(A_single).T.mean(axis=1))
            
            # save summary stats
            sumstats.append(pd.Series({
                'cluster_id': label,
                'cluster_size': len(S_single),
                'S_mean_std': np.array(S_single).T.std(axis=1).mean(),
                'A_mean_std': np.array(A_single).T.std(axis=1).mean()
            }))
            
        # prepare output
        self.S = np.stack(S).T
        self.A = np.stack(A).T
        self.clustering_stats_ = pd.concat(sumstats, axis=1).T
        
        
    def _cluster_components(self):
        """
        Example
        -------
        X = sampledata
        rica = RobustICA(n_jobs=10, n_components=5, robust_method='DBSCAN', robust_kws={'min_samples':5, 'n_jobs':10})
        rica._iterate_ica(X.values)
        rica._cluster_components()
        rica.S.shape
        rica.A.shape
        rica.clustering_stats_
        """
        # cluster
        self.clustering.fit(self.S_all.T) # ICs are in columns; we need to transpose
        # get centroids
        self._compute_centroids()
        
    
    def fit(self, X):
        # run ICA many times
        self._iterate_ica(X)
        # cluster components
        self._cluster_components()
    
    
    def transform(self, X):
        return self.S, self.A
        
        
    def fit_transform(self, X):
        self.fit(X)
        S, A = self.transform(X)
        return S, A