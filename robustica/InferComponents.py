#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
#

from sklearn.decomposition import PCA
import numpy as np


class InferComponents:
    """
    Estimate the number of principal components needed to explain a certain 
    amount of variance using `sklearn.decomposition.PCA`.
    
    Parameters
    ----------
    max_variance_explained_ratio : float, default=0.8
        Threshold of maximum variance explained by the desired number of components.
    
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.
        Hence, the None case results in::
            n_components == min(n_samples, n_features) - 1
            
    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.
        
    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
        
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
            
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
            
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
            
        If randomized :
            run randomized SVD by the method of Halko et al.
        .. versionadded:: 0.18.0
        
    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).
        .. versionadded:: 0.18.0
        
    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).
        .. versionadded:: 0.18.0
        
    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
        .. versionadded:: 0.18.0
    
    Attributes
    ----------
    pca : instance of `sklearn.decomposition.PCA`
    
    cumsum_ : np.array of length n_components
        Cumulative explained variance ratio.
        
    inferred_components_ : int
        Number of components required to explain max_variance_explained_ratio 
        amount of variance.
    
    Examples
    --------
    from robustica.examples import make_sampledata
    from robustica import InferComponents
    
    X = make_sampledata(200, 50)
    ncomp = InferComponents().fit_predict(X)
    ncomp
    """

    def __init__(
        self,
        max_variance_explained_ratio=0.8,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        # PCA parameters
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        # inference
        self.max_variance_explained_ratio = max_variance_explained_ratio

        # initialize PCA
        self.pca = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

    def fit(self, X):
        """
        Run PCA and get neccessary number of components to explain as much 
        variance as defined by max_variance_explained_ratio.
        
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            Data input.
            
        Returns
        -------
        self
        """
        self.pca.fit(X)
        self.cumsum_ = np.cumsum(self.pca.explained_variance_ratio_)
        self.inferred_components_ = np.min(
            np.where(self.cumsum_ >= self.max_variance_explained_ratio)[0]
        )

    def predict(self):
        """
        After having run `self.fit(X)`, returns `self.inferred_components_`
        
        Parameters
        ----------
        self
        
        Returns
        -------
        self.inferred_components_ : int
            Number of components required to explain max_variance_explained_ratio 
            amount of variance.
        """
        return self.inferred_components_

    def fit_predict(self, X):
        """
        Run PCA and get neccessary number of components to explain as much 
        variance as defined by max_variance_explained_ratio and returns the
        inferred number of components.
        
        Parameters
        ----------
        self
        
        Returns
        -------
        inferred_components : int
            Number of components required to explain max_variance_explained_ratio 
            amount of variance.
        """
        self.fit(X)
        return self.predict()
