#
# Author: Miquel Anglada Girotto
# Contact: miquelangladagirotto [at] gmail [dot] com
# Last Update: 2021-03-03
# 

from robustica.examples import sampledata
from sklearn.decomposition import PCA

class InferComponents():
    """
    Estimate the number of principal components needed to explain a certain 
    amount of variance.
    """
    