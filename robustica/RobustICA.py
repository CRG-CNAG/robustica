#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
#

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_samples
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import *
from sklearn_extra.cluster import *
import time


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
    
    
    def compute_distance(self, X):
        return 1 - np.abs(np.corrcoef(X))
    
    
    def fit(self, X):
        # compute dissimilarity matrix
        D = self.compute_distance(X)

        # cluster
        self.clustering.fit(D)
        self.labels_ = self.clustering.labels_


def corrmats(X, Y):
    """
    Correlation matrix between rows in X and rows in Y.
    """
    X_cent = X - X.mean(axis=1).reshape(-1, 1)
    Y_cent = Y - Y.mean(axis=1).reshape(-1, 1)

    num = X_cent.dot(Y_cent.T)
    den = np.sqrt(
        (X_cent ** 2).sum(axis=1).reshape(-1, 1)
        * (Y_cent ** 2).sum(axis=1).reshape(1, -1)
    )
    r = num / den
    return r


def compute_iq(X, labels, precomputed=False):
    """
    Compute cluster quality index.
    """
    # is X already a correlation matrix?
    if precomputed:
        correl = 1 - X
    else:
        correl = np.corrcoef(X.T)
    
    # compute Iq for every cluster
    iqs = []
    for label in np.unique(labels):
        idx_cluster = labels == label

        avg_in = correl[idx_cluster, :][:, idx_cluster].mean()
        avg_out = correl[idx_cluster, :][:, ~idx_cluster].mean()
        iq_cluster = avg_in - avg_out
        iqs.append(iq_cluster)

    df = pd.DataFrame({"cluster_id": np.unique(labels), "iq": iqs})

    return df


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
        robust_runs=100,
        robust_infer_signs=False,
        robust_method="AgglomerativeClustering",
        robust_kws={},
        robust_dimreduce=False,
    ):

        # init parameters
        ## FastICA
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
        ## robust procedure
        self.n_jobs = n_jobs
        self.robust_runs = robust_runs
        self.robust_infer_signs = robust_infer_signs
        self.robust_method = robust_method
        self.robust_kws = robust_kws
        self.robust_dimreduce = robust_dimreduce
        
        # init FastICA
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

        # init robust procedure
        ## dimension reduction (PCA) before clustering
        if self.robust_dimreduce:
            self.robust_dimreduce = "PCA"
            self.robust_dimreduce_kws = {"n_components": self.n_components}
            self._prep_dimreduce_class()
            self.dimreduce = self.dimreduce_class(**self.robust_dimreduce_kws)
            
        ## clustering algorithm
        ### decide using our default classes and parameters
        if (isinstance(robust_method, str)) & (len(robust_kws)==0):
            self.robust_kws = self._get_clustering_defaults(
                self.robust_method, self.n_components, self.robust_runs
            )
        ### init clustering class
        self._prep_cluster_class()
        self.clustering = self.cluster_class(**self.robust_kws)

    def _get_clustering_defaults(self, method, n_components, iterations):
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
            "CommonNNClustering": {"min_samples": int(iterations * 0.5)}
        }
        kws = clustering_defaults[method]
        return kws

    def _prep_dimreduce_class(self):
        if isinstance(self.robust_dimreduce, str):
            self.dimreduce_class = eval(self.robust_dimreduce)
        else:
            self.dimreduce_class = None

    def _prep_cluster_class(self):
        if isinstance(self.robust_method, str):
            self.cluster_class = eval(self.robust_method)
        else:
            self.cluster_class = self.robust_method

    def _run_ica(self, X):
        start_time = time.time()
        S = self.ica.fit_transform(X)
        A = self.ica.mixing_
        seconds = time.time() - start_time
        #(P.R. pending) convergence = self.ica.convergence_
        #(P.R. pending) n_iter = self.ica.n_iter_
        return {
            "S": S,
            "A": A,
            "time": seconds
        #(P.R. pending)     "convergence": convergence,
        #(P.R. pending)     "n_iter": n_iter,
        }

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
        print("Running FastICA multiple times...")
        # iterate
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_ica)(args)
            for args in tqdm([X for it in range(self.robust_runs)])
        )

        # prepare output
        S_all = np.hstack([r["S"] for r in result])
        A_all = np.hstack([r["A"] for r in result])
        time = {i: r["time"] for i, r in enumerate(result)}
        #(P.R. pending) convergence = {i: r["convergence"] for i, r in enumerate(result)}
        #(P.R. pending) n_iter = {i: r["n_iter"] for i, r in enumerate(result)}

        return S_all, A_all, time #(P.R. pending) convergence, n_iter,

    def _infer_components_signs(self, S_all, n_components, iterations):
        """
        Correct direction of every iteration of ICA with respect to the first one.
        """
        # components from first run are the reference
        S0 = S_all[:, 0:n_components]

        # correlate best component with the rest of iterations to decide signs
        signs = np.full(S_all.shape[1], np.nan)
        signs[0:n_components] = 1
        for it in range(1, iterations):
            start = n_components * it
            end = start + n_components
            S_it = S_all[:, start:end]

            correl = corrmats(S0.T, S_it.T)
            rows_oi = np.abs(correl).argmax(axis=0)
            cols_oi = np.arange(correl.shape[1])
            best_correls = correl[(rows_oi, cols_oi)]
            signs[start:end] = np.sign(best_correls)

        return signs

    def _compute_centroids(self, S_all, A_all, labels):
        """
        Based on https://github.com/SBRG/precise-db/blob/master/scripts/cluster_components.py
        """
        # put clusters together
        S = []
        A = []
        S_std = []
        A_std = []
        sumstats = []
        orientation = np.full(S_all.shape[1], np.nan)
        for label in np.unique(labels):
            # subset
            idx = labels == label
            S_clust = S_all[:, idx]
            A_clust = A_all[:, idx]

            # first item is base component
            Svec0 = S_clust[:, 0]
            Avec0 = A_clust[:, 0]

            # Make sure base component is facing positive
            if abs(min(Svec0)) > max(Svec0):
                Svec0 = -Svec0
                Avec0 = -Avec0
                ori = [-1]
            else:
                ori = [1]

            S_single = [Svec0]
            A_single = [Avec0]

            # Add in rest of components
            for j in range(1, S_clust.shape[1]):
                Svec = S_clust[:, j]
                Avec = A_clust[:, j]
                if pearsonr(Svec, Svec0)[0] > 0:
                    S_single.append(Svec)
                    A_single.append(Avec)
                    ori.append(1)
                else:
                    S_single.append(-Svec)
                    A_single.append(-Avec)
                    ori.append(-1)

            S_single = np.array(S_single).T
            A_single = np.array(A_single).T

            # save centroids
            S.append(S_single.mean(axis=1))
            A.append(A_single.mean(axis=1))

            # save stds
            S_std.append(np.array(S_clust).std(axis=1))
            A_std.append(np.array(A_clust).std(axis=1))

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

            # save orientation
            orientation[idx] = ori

        # prepare output
        S = np.stack(S).T
        A = np.stack(A).T
        S_std = np.stack(S_std).T
        A_std = np.stack(A_std).T
        clustering_stats = pd.concat(sumstats, axis=1).T

        return S, A, S_std, A_std, clustering_stats, orientation

    def _compute_robust_components(self, S_all):
        """
        Example
        -------
        X = sampledata
        rica = RobustICA(n_jobs=10, n_components=5, robust_method='DBSCAN', robust_kws={'min_samples':5, 'n_jobs':10})
        rica._iterate_ica(X.values)
        rica._cluster_components()
        rica.S.shape
        rica.A.shape
        rica.clustering.stats_
        """
        ## infer signs of components across ICA runs
        if self.robust_infer_signs:
            print("Inferring sign of components...")
            signs = self.infer_components_signs(
                S_all, self.n_components, self.robust_runs
            )
        else:
            signs = np.ones(S_all.shape[1])

        Y = S_all * signs

        ## compress feature space
        if self.robust_dimreduce != False:
            print("Reducing dimensions...")
            Y = self.dimreduce.fit_transform(Y.T).T

        ## cluster
        self.clustering.fit(Y.T)
        labels = self.clustering.labels_

        ## compute robust components
        S, A, S_std, A_std, clustering_stats, orientation = self._compute_centroids(
            S_all * signs, A_all * signs, labels
        )
        
        return S, A, S_std, A_std, clustering_stats, signs, orientation

    def fit(self, X):
        # run ICA multiple times
        ## iterate
        S_all, A_all, time = self._iterate_ica(X) #(P.R. pending)

        ## save attributes
        self.S_all = S_all
        self.A_all = A_all
        self.time = time
        #(P.R. pending) self.convergence_ = convergence
        #(P.R. pending) self.n_iter_ = n_iter

        # Compute robust independent components
        S, A, S_std, A_std, clustering_stats, signs, orientation = _compute_robust_components(S_all)

        ## save attributes
        self.S = S
        self.A = A
        self.S_std = S_std
        self.A_std = A_std
        self.clustering.stats_ = clustering_stats
        self.signs_ = signs
        self.orientation_ = orientation

    def transform(self, X):
        return self.S, self.A

    def fit_transform(self, X):
        self.fit(X)
        S, A = self.transform(X)
        return S, A

    def evaluate_clustering(self, silhouette_metric='euclidean'):
        """
        Run after fit()
        """
        S_all = self.S_all
        labels = self.clustering.labels_
        sign = self.signs_
        orientation = self.orientation_

        # prep
        Y = (S_all * sign * orientation).T

        # silhouette of components
        self.clustering.silhouette_scores_ = silhouette_samples(
            Y, labels, metric=silhouette_metric
        )

        # Iq of components
        self.clustering.iq_scores_ = compute_iq(Y, labels)

        # update clustering stats
        self.clustering.stats_["mean_silhouette"] = [
            np.mean(self.clustering.silhouette_samples_[labels == cluster_id])
            for cluster_id in self.clustering.stats_["cluster_id"]
        ]

        self.clustering.stats_ = pd.merge(
            self.clustering.stats_, self.clustering.iq_scores_, on="cluster_id"
        )

#(P.R. pending) 
#     def prepare_summary(self):
#         """
#         Run after fit()
#         """
#         df = pd.DataFrame.from_dict(self.convergence_, orient="index").T.melt().dropna()
#         df.columns = ["iteration_robustica", "convergence_score"]
#         df["iteration_ica"] = df.groupby("iteration_robustica").cumcount()
#         df = df.join(pd.Series(self.time_, name="time_ica"), on="iteration_robustica")
#         df = df.join(
#             pd.Series(self.n_iter_, name="convergence_n_iter"), on="iteration_robustica"
#         )
#         df["max_iter"] = self.ica.max_iter
#         df["tol"] = self.ica.tol
#         return df

