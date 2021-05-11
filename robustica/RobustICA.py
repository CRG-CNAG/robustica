#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
#

from robustica.examples import sampledata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
import multiprocessing as mp
from scipy import stats, sparse
from scipy.stats import pearsonr
from sklearn.cluster import *
from sklearn_extra.cluster import *
import time


class Sastry:
    """
    Special distance matrix with precomputed DBSCAN.
    """

    def __init__(self, **kws):
        self.clustering = DBSCAN(metric="precomputed", **kws)

    def fit(self, X):
        # compute similarity matrix
        dist = abs(np.dot(X, X.T))  # the input will be transposed
        dist[dist < 0.5] = 0
        D = 1 - dist

        # correct floating point imprecision
        D[D < 0] = 0
        D[D > 1] = 0

        # cluster
        self.clustering.fit(D)
        self.labels_ = self.clustering.labels_


class Icasso:
    """
    distance_threshold = 
    
    Example
    -------
    X = sampledata
    clustering = Icasso(distance_threshold=0.8)
    ={'distance_threshold':1.2,'n_clusters':None}
    """

    def __init__(self, **kws):
        self.clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="average", **kws
        )

    def fit(self, X):
        # compute dissimilarity matrix
        corr = np.abs(np.corrcoef(X))
        D = 1 - corr

        # correct floating point imprecision
        D[D < 0] = 0
        D[D > 1] = 0

        # continue
        D = np.sqrt(D)

        # cluster
        self.clustering.fit(D)
        self.labels_ = self.clustering.labels_
        

class RobustICA:
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

    def __init__(
        self,
        n_components=None,
        algorithm="parallel",
        whiten=True,
        fun="logcosh",
        fun_args=None,
        max_iter=200,
        tol=0.0001,
        w_init=None,
        random_state=None,
        n_jobs=None,
        robust_iter=100,
        robust_method="DBSCAN",
        robust_kws="auto",
        robust_dimreduce=True,
    ):

        # parameters for FastICA
        if max_iter < 1:
            raise ValueError(
                "max_iter should be greater than 1, got "
                "(max_iter={})".format(max_iter)
            )
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
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            max_iter=self.max_iter,
            tol=self.tol,
            w_init=self.w_init,
            random_state=self.random_state,
        )

        # parameters for robust procedure
        self.n_jobs = n_jobs
        self.robust_iter = robust_iter
        self.robust_method = robust_method
        if robust_kws == "auto":
            self.robust_kws = self._get_defaults(
                self.robust_method, self.n_components, self.robust_iter
            )
        else:
            self.robust_kws = robust_kws
        if robust_dimreduce:
            self.robust_dimreduce = "PCA"
            self.robust_dimreduce_kws = {"n_components": self.n_components}

        # initialize dimension reduction function
        if self.robust_dimreduce is not None:
            self._prep_dimreduce_func()
            self.dimreduce = self.dimreduce_func(**self.robust_dimreduce_kws)

        # initialize clustering function
        self._prep_cluster_func()
        self.clustering = self.cluster_func(**self.robust_kws)

    def _get_defaults(self, method, n_components, iterations):
        clustering_defaults = {
            # sklearn
            "AffinityPropagation": {},
            "AgglomerativeClustering": {"n_clusters": n_components},
            "Birch": {"n_clusters": n_components},
            "DBSCAN": {"min_samples": int(iterations * 0.5)},
            "FeatureAgglomeration": {"n_clusters": n_components},
            "KMeans": {"n_clusters": n_components},
            "MiniBatchKMeans": {"n_clusters": n_components},
            "MeanShift": {},
            "OPTICS": {"min_samples": int(iterations * 0.5)},
            "SpectralClustering": {"n_clusters": n_components},
            "SpectralBiclustering": {"n_clusters": n_components},
            "SpectralCoclustering": {"n_clusters": n_components},
            # sklearn_extra
            "KMedoids": {"n_clusters": n_components},
            "CommonNNClustering": {"min_samples": int(iterations * 0.5)},
        }
        kws = clustering_defaults[method]
        return kws

    def _run_ica(self, X):
        start_time = time.time()
        S = self.ica.fit_transform(X)
        A = self.ica.mixing_
        convergence = self.ica.convergence_
        n_iter = self.ica.n_iter_
        seconds = time.time() - start_time
        return {"S": S, "A": A, "convergence": convergence, "n_iter": n_iter, "time": seconds}

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
        self.S_all = np.hstack([r["S"] for r in result])
        self.A_all = np.hstack([r["A"] for r in result])
        self.convergence_ = {i: r["convergence"] for i, r in enumerate(result)}
        self.n_iter_ = {i: r["n_iter"] for i, r in enumerate(result)}
        self.time_ = {i: r["time"] for i, r in enumerate(result)}

    def _prep_dimreduce_func(self):
        if isinstance(self.robust_dimreduce, str):
            self.dimreduce_func = eval(self.robust_dimreduce)
        else:
            self.dimreduce_func = None

    def _prep_cluster_func(self):
        if isinstance(self.robust_method, str):
            self.cluster_func = eval(self.robust_method)
        else:
            self.cluster_func = self.robust_method

    def _get_iteration_signs(self, X):
        """
        Correct direction of every iteration of ICA with respect to the first one.
        """
        # init
        S_all = self.S_all
        A_all = self.A_all
        n_components = self.n_components
        iterations = self.robust_iter

        # get component that explains the most variance of X from first iteration
        S0 = S_all[:, 0:n_components]
        A0 = A_all[:, 0:n_components]
        tss = []
        for i in range(n_components):
            pred = np.dot(S0[:, i].reshape(-1, 1), A0[:, i].reshape(1, -1))
            tss.append(np.sum((X - pred) ** 2))  # total sum of squares
        best_comp = S0[:, np.argmax(tss)]

        # correlate best component with the rest of iterations to decide signs
        signs = np.full(S_all.shape[1], np.nan)
        signs[0:n_components] = 1
        for it in range(1, iterations):
            start = n_components * it
            end = start + n_components
            S_it = S_all[:, start:end]

            correl = np.apply_along_axis(
                lambda x: pearsonr(x, best_comp)[0], axis=0, arr=S_it
            )
            best_correl = correl[np.argmax(np.abs(correl))]
            signs[start:end] = np.sign(best_correl)

        return signs

    def _compute_centroids(self):
        # correct signs
        signs = self.iteration_signs_
        S_all = (signs * self.S_all).T
        A_all = (signs * self.A_all).T

        # put clusters together
        S = []
        A = []
        S_std = []
        A_std = []
        sumstats = []
        labels = self.clustering.labels_
        for label in np.unique(labels):
            # subset
            idx = np.where(labels == label)[0]
            S_clust = S_all[idx, :]
            A_clust = A_all[idx, :]

            # save centroids
            S.append(np.array(S_clust).T.mean(axis=1))
            A.append(np.array(A_clust).T.mean(axis=1))
            
            # save stds
            S_std.append(np.array(S_clust).T.std(axis=1))
            A_std.append(np.array(A_clust).T.std(axis=1))
            
            # save summary stats
            sumstats.append(
                pd.Series(
                    {
                        "cluster_id": label,
                        "cluster_size": len(S_clust),
                        "S_mean_std": np.array(S_clust).T.std(axis=1).mean(),
                        "A_mean_std": np.array(A_clust).T.std(axis=1).mean(),
                    }
                )
            )

        # prepare output
        self.S = np.stack(S).T
        self.A = np.stack(A).T
        self.S_std = np.stack(S_std).T
        self.A_std = np.stack(A_std).T
        self.clustering_stats_ = pd.concat(sumstats, axis=1).T

    def _cluster_components(self, X):
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
        # correct signs of components by iteration
        print("Correcting component signs across iterations...")
        self.iteration_signs_ = self._get_iteration_signs(X)
        S_all = self.iteration_signs_ * self.S_all
        S_all = S_all.T  # ICs are in columns; we need to transpose

        # reduce dimensions
        if self.robust_dimreduce is not None:
            print("Reducing dimensions...")
            S_all = self.dimreduce.fit_transform(S_all)

        # cluster
        print("Clustering...")
        self.clustering.fit(S_all)

        # get centroids
        self._compute_centroids()

    def fit(self, X):
        # run ICA many times
        self._iterate_ica(X)
        # cluster components
        self._cluster_components(X)

    def transform(self, X):
        return self.S, self.A

    def fit_transform(self, X):
        self.fit(X)
        S, A = self.transform(X)
        return S, A
    
    def prepare_summary(self):
        df = pd.DataFrame.from_dict(self.convergence_, orient='index').T.melt().dropna()
        df.columns = ['iteration_robustica','convergence_score']
        df['iteration_ica'] = df.groupby('iteration_robustica').cumcount()
        df = df.join(pd.Series(self.time_, name='time_ica'), on='iteration_robustica')
        df = df.join(pd.Series(self.n_iter_, name='convergence_n_iter'), on='iteration_robustica')
        df['max_iter'] = self.ica.max_iter
        df['tol'] = self.ica.tol
        return df
