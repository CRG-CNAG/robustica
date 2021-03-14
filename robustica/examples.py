import pandas as pd
import numpy as np

# sampledata
nrow, ncol = 20000, 400
sampledata = pd.DataFrame(np.random.rand(nrow, ncol), 
                          index=['row%s'%r for r in range(nrow)], 
                          columns=['col%s'%c for c in range(ncol)])