import pandas as pd
import numpy as np

# sampledata
nrow, ncol = 100, 20
sampledata = pd.DataFrame(np.random.rand(100, 20), 
                          index=['row%s'%r for r in range(nrow)], 
                          columns=['col%s'%c for c in range(ncol)])