# `robustica`
Fully cumstomizable robust Independent Components Analysis (ICA).

[![pipy](https://img.shields.io/pypi/v/robustica?color=informational)](https://pypi.python.org/pypi/robustica)
[![conda](https://anaconda.org/conda-forge/robustica/badges/version.svg)](https://anaconda.org/conda-forge/robustica)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Description
This package contains 3 modules:
- `RobustICA`

    Defines the most important class that allows to perform and customize robust independent component analysis.
    
- `InferComponents`

    Retrieves the number of components that explain a user-defined percentage of variance.

- `examples`
    
    Contains handy functions to quickly create or access example datasets.

## Requirements
- `numpy` (1.19.2)
- `pandas` (1.1.2)
- `scipy` (1.6.2)
- `scikit-learn` (0.23.2)
- `scikit-learn-extra` (0.2.0)
- `joblib` (1.0.1)
- `tqdm` (4.59.0)

## Installation
### conda
```shell
conda install -c conda-forge robustica
```
### pip
```
pip install robustica
```

## Usage
```python
from robustica import RobustICA
from robustica.examples import make_sampledata

X = make_sampledata(ncol=300, nrow=2000, seed=123)

rica = RobustICA(n_components=10)
S, A = rica.fit_transform(X.values)
```

## Tutorials
- [Basic pipeline for exploratory analysis]()
- [Using a custom clustering class]()


## Contributors
- Miquel Anglada Girotto ([![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/m1quelag.svg?style=social&label=Follow%20%40m1quelag)](https://twitter.com/m1quelag))

## References
