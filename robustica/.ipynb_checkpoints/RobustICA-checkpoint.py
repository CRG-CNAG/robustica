# 2020 - Centre de Regulacio Genomica (CRG) - All Rights Reserved
#
# Author: Miquel Anglada Girotto
# Contact: miquel [dot] anglada [at] crg [dot] eu
# Last Update: 2022-06-26
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


def abs_pearson_dist(X):
    """Compute Pearson dissimilarity between columns.
    
    Parameters
    ----------
    X: np.array of shape (n_features, n_samples)
        Input data.
        
    Returns
    -------
    D: np.array of shape (n_samples, n_samples)
        Dissimilarity matrix.
        
    Examples
    --------
    .. code-block:: python
    
        from robustica import abs_pearson_dist
        from robustica.examples import make_sampledata

        X = make_sampledata(15, 5)
        D = abs_pearson_dist(X)
        D.shape
    """
    D = np.clip((1 - np.abs(np.corrcoef(X.T))), 0, 1)
    return D


def corrmats(X, Y):
    """
    Vectorized implementation of pairwise correlations between rows in X and rows in Y.
    Make sure that the number of columns in X and Y is the same.
    
    Parameters
    ----------
    X : np.array of shape (n_features_x, n_samples_x)
    
    Y : np.array of shape (n_features_y, n_samples_y)
    
    Returns
    -------
    r : np.array of shape (n_features_x, n_features_y)
    
    Examples
    --------
    .. code-block:: python
    
        from robustica import corrmats
        from robustica.examples import make_sampledata

        X = make_sampledata(15, 5)
        Y = make_sampledata(20, 5)
        r = corrmats(X, Y)
        r.shape
    """
    assert X.shape[1] == Y.shape[1]

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
    Compute cluster index of quality as suggested by Himberg, J., & Hyvarinen (2004) (DOI: https://doi.org/10.1109/NNSP.2003.1318025).
    This method requires computing a square correlation matrix.
    
    Parameters
    ----------
    X : np.array of shape (n_samples, n_features)
    
    labels: list or np.array
        Clustering labels indicating to which cluster every observation belongs.
    
    precomputed: bool, default=False
        Indicates whether X is a square pairwise correlation matrix.
    
    
    Returns
    -------
    df : pd.DataFrame
        Dataframe with cluster labels ('cluster_id') and their corresponding Iq scores.
    
    Examples
    --------
    .. code-block:: python
    
        from robustica import compute_iq
        from robustica.examples import make_sampledata

        X = make_sampledata(5, 15)
        labels = [1,1,2,1,2]
        df = compute_iq(X, labels)
        df
    """
    # is X already a correlation matrix?
    if precomputed:
        assert X.shape[0] == X.shape[1]
        correl = 1 - X
    else:
        correl = np.corrcoef(X)

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
    r"""
    Class to perform robust Independent Component Analysis (ICA) using different
    methods to cluster together the independent components computed via 
    `sklearn.decomposition.FastICA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html>`_. 
    
    By default, it carries out the *Icasso* algorithm
    using aglomerative clustering with average linkage and a precomputed Pearson
    dissimilarity matrix.
    
    Schematically, RobustICA works like this:
        1) Run ICA multiple times and save source (S) and mixing (A) matrices.
        2) Cluster the components into robust components using Ss across runs.
            2.1) If we use a precomputed dissimilarity:
                2.1.1) Precompute dissimilarity
            2.2) If we don't use a precomputed dissimilarity:
                2.2.1) (Optional) Infer and correct component signs across runs
                2.2.2) (Optional) Reduce the feature space with PCA
            2.3) Cluster components across all S runs
            2.4) Use clustering labels to compute the centroid of each cluster, i.e. the robust component in both S and A.
            
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
        See `Glossary <https://scikit-learn.org/stable/glossary.html#term-random_state>`_.
    
    robust_runs : int, default=100
        Number of times to run FastICA.
        
    robust_infer_signs : bool, default=True
        If robust_infer_signs is True, we infer and correct the signs of components 
        across ICA runs before clustering them.
        
    robust_method : str or callable, default="DBSCAN"
        Clustering class to compute robust components across ICA runs.
        If str, choose one of the following clustering algorithms from 
        `sklearn.cluster`:
            - "AgglomerativeClustering"
            - "AffinityPropagation"
            - "Birch"
            - "DBSCAN"
            - "FeatureAgglomeration"
            - "KMeans"
            - "MiniBatchKMeans"
            - "MeanShift"
            - "OPTICS"
            - "SpectralClustering"
            
        or from `sklearn_extra.cluster`:
            - "KMedoids"
            - "CommonNNClustering"
            
        If class, the algorithm expects a clustering class with a `self.fit()` method 
        that creates a `self.clustering.labels_` attribute returning the list of clustering labels.
        
    robust_kws : dict, default={"linkage": "average"}
        Keyword arguments to send to clustering class defined by robust_method.
        If robust_method is str and if "n_clusters" or "min_samples" are not 
        defined in robust_kws, robust_kws will be updated with either 
        {"n_clusters": self.n_components} or 
        {"min_samples": int(self.robust_runs * 0.5)} accordingly.
        
    robust_dimreduce : bool, default=True
        If robust_dimreduce is True, we use `sklearn.decomposition.PCA` with 
        the same n_components to reduce the feature space across ICA runs after 
        sign inference and correction (if robust_infer_signs=True) and before clustering.
        
    robust_precompdist_func :  "abs_pearson_dist" or callable, default="abs_pearson_dist"
        If robust_kws contain the value "precomputed", we precompute a distance
        matrix by executing robust_precomp_dist_func and use it for clustering.
        
    Attributes
    ----------
    S : np.array of shape (n_features, n_components)
        Robust source matrix computed using the centroids of every cluster.
    
    A : np.array of shape (n_samples, n_components)
        Robust mixing matrix computed using the centroids of every cluster.
        
    S_std : np.array of shape (n_features, n_components)
        Within robust component standard deviation across features.
    
    A_std : np.array of shape (n_features, n_components)
        Within robust component standard deviation across samples.
        
    S_all : np.array of shape (n_features, n_components * robust_runs)
        Concatenated source matrices corresponding to every run of ICA. 
        
    A_all : np.array of shape (n_features, n_components * robust_runs)
        Concatenated mixing matrices corresponding to every run of ICA.

    time : dict of length n_components * robust_runs
        Time to execute every run of ICA for robust_runs times. Dictionary 
        structured as {run : seconds}.
    
    signs_ : np.array of length n_components * robust_runs
        Array of positive or negative ones used to correct for signs before 
        clustering.
    
    orientation_ : np.array of length n_components * robust_runs
        Array of positive or negative ones used to orient labeled components
        after clustering so that largest weights face positive.
    
    clustering : class instance
        Instance used to cluster components in S_all across ICA runs. The clustering
        labels can be found in the attribute `self.clustering.labels_`. 
        In `self.clustering.stats_` you can find information on cluster sizes and
        mean standard deviations per cluster in both S and A robust matrices.
    
    Examples
    --------
    .. code-block:: python
    
        from robustica import RobustICA
        from robustica.examples import make_sampledata

        X = make_sampledata(200,50)
        rica = RobustICA(n_components=10)
        S, A = rica.fit_transform(X)
    
    Notes
    -----
    Icasso procedure based on
    *Himberg, J., & Hyvarinen, A. "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization". 
    IEEE XIII Workshop on Neural Networks for Signal Processing (2003).* 
    DOI: https://doi.org/10.1109/NNSP.2003.1318025
    
    Centroid computation based on
    *Sastry, Anand V., et al. "The Escherichia coli transcriptome mostly 
    consists of independently regulated modules." 
    Nature communications 10.1 (2019): 1-14.*
    DOI: https://doi.org/10.1038/s41467-019-13483-w
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
        robust_infer_signs=True,
        robust_dimreduce=True,
        robust_method="DBSCAN",
        robust_kws={},
        robust_precompdist_func="abs_pearson_dist",
        verbose=True
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
        self.robust_precompdist_func = robust_precompdist_func
        self.verbose = verbose
        
        # reproducibility: generate random state for each iteration
        if self.random_state is not None:
            if isinstance(self.random_state, int): 
                rng = np.random.RandomState(self.random_state)
                
            self.random_states = rng.randint(0, self.robust_runs*100, size=self.robust_runs)
        else:
            self.random_states = [None for it in range(self.robust_runs)]
        
        # init robust procedure
        ## dimension reduction (PCA) before clustering
        if self.robust_dimreduce:
            self.robust_dimreduce = "PCA"
            self.robust_dimreduce_kws = {"n_components": self.n_components}
            self._prep_dimreduce_class()
            self.dimreduce = self.dimreduce_class(**self.robust_dimreduce_kws)

        ## precompute distance matrix
        self._prep_precompdist_func()

        ## clustering algorithm
        ### decide using our default classes and parameters
        self._set_clustering_defaults()

        ### init clustering class
        self._prep_clustering_class()
        self.clustering = self.clustering_class(**self.robust_kws)

    def _set_clustering_defaults(self):
        no_precomputed = [
            "Brich",
            "KMeans",
            "MiniBatchKMeans",
            "Meanshift",
            "SpectralClustering",
        ]
        clustering_defaults = {
            # sklearn
            "AffinityPropagation": {},
            "AgglomerativeClustering": {"n_clusters": self.n_components},
            "Birch": {"n_clusters": self.n_components},
            "DBSCAN": {"min_samples": int(self.robust_runs * 0.5)},
            "FeatureAgglomeration": {"n_clusters": self.n_components},
            "KMeans": {"n_clusters": self.n_components},
            "MiniBatchKMeans": {"n_clusters": self.n_components},
            "MeanShift": {},
            "OPTICS": {"min_samples": int(self.robust_runs * 0.5)},
            "SpectralClustering": {"n_clusters": self.n_components},
            # sklearn_extra
            "KMedoids": {"n_clusters": self.n_components},
            "CommonNNClustering": {"min_samples": int(self.robust_runs * 0.5)},
        }
        if (isinstance(self.robust_method, str)) & (
            not any(
                np.isin(["n_clusters", "min_samples"], list(self.robust_kws.keys()))
            )
        ):
            if self.verbose:
                print('Setting clustering defaults:', clustering_defaults[self.robust_method])
            self.robust_kws = {
                **self.robust_kws,
                **clustering_defaults[self.robust_method],
            }

    def _prep_dimreduce_class(self):
        if isinstance(self.robust_dimreduce, str):
            self.dimreduce_class = eval(self.robust_dimreduce)
        else:
            self.dimreduce_class = None

    def _prep_precompdist_func(self):
        if isinstance(self.robust_precompdist_func, str):
            self.robust_precompdist_func = eval(self.robust_precompdist_func)

    def _prep_clustering_class(self):
        if isinstance(self.robust_method, str):
            self.clustering_class = eval(self.robust_method)
        else:
            self.clustering_class = self.robust_method

    def _run_ica(self, X, random_state):
        """
        Execute and instance of `sklearn.decomposition.FastICA` once.
        
        Parameters
        ----------
        X : np.array of shape (n_features, n_samples)
            Data input.
        
        Returns
        -------
        output : dict
            Dictionary containing outputs from the run: source matrix ("S"),
            mixing matrix ("A") and execution time in seconds ("time")
        """

        start_time = time.time()
        # init FastICA
        ica = FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            max_iter=self.max_iter,
            tol=self.tol,
            w_init=self.w_init,
            random_state=random_state,
        )
        S = ica.fit_transform(X)
        A = ica.mixing_
        seconds = time.time() - start_time
        output = {"S": S, "A": A, "time": seconds}
        return output

    def _iterate_ica(self, X):
        """
        Execute and instance of `sklearn.decomposition.FastICA` for robust_runs
        times with random initialisations.
        
        Parameters
        ----------
        X : np.array of shape (n_features, n_samples)
            Data input.
        
        Returns
        -------
        S_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated source matrices corresponding to every run of ICA. 

        A_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated mixing matrices corresponding to every run of ICA.

        time : dict of length n_components * robust_runs
            Time to execute every run of ICA for robust_runs times. Dictionary 
            structured as {run : seconds}.            
        """
        
        if self.verbose:
            print("Running FastICA multiple times...")
            
        # iterate
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_ica)(X, random_state)
            for random_state in tqdm(self.random_states, disable=(not self.verbose))
        )

        # prepare output
        S_all = np.hstack([r["S"] for r in result])
        A_all = np.hstack([r["A"] for r in result])
        time = {i: r["time"] for i, r in enumerate(result)}

        return S_all, A_all, time

    def _infer_components_signs(self, S_all, n_components, robust_runs):
        """
        Correct direction of every iteration of ICA with respect to the first one.
        
        Parameters
        ----------
        S_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated source matrices corresponding to every run of ICA. 
            
        n_components : int, default=None
            Number of components to use. If None is passed, all are used.
            
        robust_runs : int, default=100
            Number of times to run FastICA.
        
        Returns
        -------
        signs : np.array of length n_components * robust_runs
            Array of positive or negative ones used to correct for signs before 
            clustering.
        """
        # components from first run are the reference
        S0 = S_all[:, 0:n_components]

        # correlate best component with the rest of iterations to decide signs
        signs = np.full(S_all.shape[1], np.nan)
        signs[0:n_components] = 1
        for it in range(1, robust_runs):
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
        Compute centroid for every cluster while re-orienting components so that
        large absolute weights face positive.
        
        Parameters
        ----------
        S_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated source matrices corresponding to every run of ICA. 

        A_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated mixing matrices corresponding to every run of ICA.        
            
        labels : array-like object of length n_components * robust_runs
            List of clustering labels.
        
        Returns
        -------
        S : np.array of shape (n_features, n_components)
            Robust source matrix computed using the centroids of every cluster.

        A : np.array of shape (n_samples, n_components)
            Robust mixing matrix computed using the centroids of every cluster.

        S_std : np.array of shape (n_features, n_components)
            Within robust component standard deviation across features.

        A_std : np.array of shape (n_features, n_components)
            Within robust component standard deviation across samples.
        
        clustering_stats : pd.DataFrame
            DataFrame with information on the cluster sizes and mean standard
            deviations across clusters for both S and A matrices. 
            This object will become available as part of the clustering 
            instance: `self.clustering.stats_`.
        
        orientation : np.array of length n_components * robust_runs
            Array of positive or negative ones used to orient labeled components
            after clustering so that largest weights face positive.

        Notes
        -----
        Based on https://github.com/SBRG/precise-db/blob/782c252d4e4e6fb7e5d0037b85b1a00b59c6f1fe/scripts/cluster_components.py#L151
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
            S_std.append(S_single.std(axis=1))
            A_std.append(A_single.std(axis=1))

            # save summary stats
            sumstats.append(
                pd.Series(
                    {
                        "cluster_id": label,
                        "cluster_size": sum(idx),
                        "S_mean_std": S_single.std(axis=1).mean(),
                        "A_mean_std": A_single.std(axis=1).mean(),
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

    def _compute_robust_components(self, S_all, A_all):
        """
        Recipe to compute robust components after running ICA multiple times.
        
        Parameters
        ----------
        S_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated source matrices corresponding to every run of ICA. 

        A_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated mixing matrices corresponding to every run of ICA.  
            
        Returns
        -------
        S : np.array of shape (n_features, n_components)
            Robust source matrix computed using the centroids of every cluster.

        A : np.array of shape (n_samples, n_components)
            Robust mixing matrix computed using the centroids of every cluster.

        S_std : np.array of shape (n_features, n_components)
            Within robust component standard deviation across features.

        A_std : np.array of shape (n_features, n_components)
            Within robust component standard deviation across samples.
        
        clustering_stats : pd.DataFrame
            DataFrame with information on the cluster sizes and mean standard
            deviations across clusters for both S and A matrices. 
            This object will become available as part of the clustering 
            instance: `self.clustering.stats_`.
            
        signs : np.array of length n_components * robust_runs
            Array of positive or negative ones used to correct for signs before 
            clustering.
        
        orientation : np.array of length n_components * robust_runs
            Array of positive or negative ones used to orient labeled components
            after clustering so that largest weights face positive.    
        """

        ## infer signs of components across ICA runs
        if self.robust_infer_signs:
            if self.verbose:
                print("Inferring sign of components...")
            signs = self._infer_components_signs(
                S_all, self.n_components, self.robust_runs
            )
        else:
            signs = np.ones(S_all.shape[1])

        Y = S_all * signs

        ## compress feature space
        if self.robust_dimreduce != False:
            if self.verbose:
                print("Reducing dimensions...")
            Y = self.dimreduce.fit_transform(Y.T).T

        ## precompute distance matrix
        contains_precomputed = np.isin(["precomputed"], list(self.robust_kws.values()))[
            0
        ]
        if contains_precomputed and (self.robust_precompdist_func is not None):
            # then, we want to use our precompdist_func
            if self.verbose:
                print("Precomputing distance matrix...")
            Y = self.robust_precompdist_func(Y)

        ## cluster
        if self.verbose:
            print("Clustering...")
        self.clustering.fit(Y.T)
        labels = self.clustering.labels_

        ## compute robust components
        if self.verbose:
            print("Computing centroids...")
        S, A, S_std, A_std, clustering_stats, orientation = self._compute_centroids(
            S_all * signs, A_all * signs, labels
        )

        return S, A, S_std, A_std, clustering_stats, signs, orientation

    def fit(self, X):
        """
        Runs ICA robust_runs times and computes robust independent components.
        
        Parameters
        ----------
        X : np.array of shape (n_features, n_samples)
            Data input.
        
        Returns
        -------
        self
        """
        # run ICA multiple times
        ## iterate
        S_all, A_all, time = self._iterate_ica(X)

        ## save attributes
        self.S_all = S_all
        self.A_all = A_all
        self.time = time

        # Compute robust independent components
        (
            S,
            A,
            S_std,
            A_std,
            clustering_stats,
            signs,
            orientation,
        ) = self._compute_robust_components(S_all, A_all)

        ## save attributes
        self.S = S
        self.A = A
        self.S_std = S_std
        self.A_std = A_std
        self.clustering.stats_ = clustering_stats
        self.signs_ = signs
        self.orientation_ = orientation

    def transform(self):
        """
        After having executed the `self.fit(X)` method, return robust S and A matrices.
        
        Parameters
        ----------
        self
        
        Returns
        -------
        S : np.array of shape (n_features, n_components)
            Robust source matrix computed using the centroids of every cluster.

        A : np.array of shape (n_samples, n_components)
            Robust mixing matrix computed using the centroids of every cluster.
        """
        return self.S, self.A

    def fit_transform(self, X):
        """
        Runs ICA robust_runs times and computes robust independent components and
        returns the robust S and A matrices.
        
        Parameters
        ----------
        X : np.array of shape (n_features, n_samples)
            Data input.
        
        Returns
        -------
        S : np.array of shape (n_features, n_components)
            Robust source matrix computed using the centroids of every cluster.

        A : np.array of shape (n_samples, n_components)
            Robust mixing matrix computed using the centroids of every cluster.
        """

        self.fit(X)
        S, A = self.transform()
        return S, A

    def evaluate_clustering(
        self, S_all, labels, signs, orientation, metric="euclidean"
    ):
        """
        After having executed the `self.fit(X)` method, computes silhouette scores
        by samples and index of quality (Iq) proposed by Himberg, J., & Hyvarinen 
        (2004) (DOI: https://doi.org/10.1109/NNSP.2003.1318025).
        
        Silhouette scores for each component are computed using 
        `sklearn.metrics.silhouette_samples`.
        
        Iq scores foreach cluster (i.e. robust component) are computed using 
        `robustica.RobustICA.compute_iq`.
        
        Parameters
        ----------
        S_all : np.array of shape (n_features, n_components * robust_runs)
            Concatenated source matrices corresponding to every run of ICA. 
            
        labels : array-like object of length n_components * robust_runs
            List of clustering labels.
        
        signs : np.array of length n_components * robust_runs
            Array of positive or negative ones used to correct for signs before 
            clustering.
        
        orientation : np.array of length n_components * robust_runs
            Array of positive or negative ones used to orient labeled components
            after clustering so that largest weights face positive. 
        
        metric : str
            Metric to use to evaluate the clustering with 
            `sklearn.metrics.silhouette_samples`. If metric='precomputed', S_all
            has to be a square matrix with a diagonal of 0s.
            
        Returns
        -------
        evaluation : pd.DataFrame
            Dataframe with information on the average silhouette scores and Iq
            for each cluster.
        
        Attributes
        ----------
        self.clustering.silhouette_scores_ : np.array of length n_components * robust_runs
            Silhouette coefficient for each component.
            
        self.clustering.iq_scores_ : np.array of length n_components * robust_runs
            Iq coefficient for each component.
        """

        # prep
        if self.verbose:
            print("Computing Silhouettes...")
        if metric == "precomputed":
            Y = S_all
        else:
            Y = (S_all * signs * orientation).T

        # silhouette of components
        self.clustering.silhouette_scores_ = silhouette_samples(
            Y, labels, metric=metric
        )

        # Iq of components
        if self.verbose:
            print("Computing Iq...")
        if metric == "precomputed":
            self.clustering.iq_scores_ = compute_iq(Y, labels, precomputed=True)
        else:
            self.clustering.iq_scores_ = compute_iq(Y, labels)

        evaluation = pd.DataFrame({"cluster_id": np.unique(labels)})

        # update clustering stats
        evaluation["mean_silhouette"] = [
            np.mean(self.clustering.silhouette_scores_[labels == cluster_id])
            for cluster_id in evaluation["cluster_id"]
        ]

        evaluation = pd.merge(evaluation, self.clustering.iq_scores_, on="cluster_id")

        return evaluation
