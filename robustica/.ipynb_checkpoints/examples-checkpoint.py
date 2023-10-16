# 2020 - Centre de Regulacio Genomica (CRG) - All Rights Reserved
#
# Author: Miquel Anglada Girotto
# Contact: miquel [dot] anglada [at] crg [dot] eu
# Last Update: 2021-07-24
#

import numpy as np

def make_sampledata(nrow, ncol, seed=None):
    """
    Prepare a random sample dataset with `np.random.rand`.
    
    Parameters
    ----------
    nrow : int
        Number of desired rows.
    ncol : int
        Number of desired columns
    seed : int, default=None
        Random seed in case we want full reproducibility.
    
    Returns
    -------
    sampledata : np.array of shape (nrow, ncol)
        Resulting random sample dataset.
        
    Example
    -------
    .. code-block:: python

        from robustica.examples import make_sampledata
        X = make_sampledata(ncol=300, nrow=2000, seed=123)
        
    """
    np.random.seed(seed=seed)
    sampledata = np.random.rand(nrow, ncol)
    
    return sampledata