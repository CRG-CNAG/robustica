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
    """
    np.random.seed(seed=seed)
    sampledata = np.random.rand(nrow, ncol)
    
    return sampledata