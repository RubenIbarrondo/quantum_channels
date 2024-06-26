# PyQch

PyQch is a package containing the functions that I frequently require to make some numerics with quantum channels.

## Requirements

PyQch uses `numpy` and `scipy`.

## Installation

The package is not indexed. You can access the code in src.

## Usage

States and channels are encoded as 2D `np.ndarray`s representing density and transition matrices, respectively.
Although it implies that to apply a channel the state has to be reshaped into a 1D array and back, the representation kept integration with `numpy` and `scipy` simplest. 

For example, the following code generates two random states and a depolarizing channel, then prints their initial and final trace distance.

```python
import numpy as np
import pyqch.random_generators as rg
import pyqch.channel_families as cf
from pyqch.divergences import tr_dist

# Set the dimension of the Hilbert space
dim = 3

# Create two random pure states
rho = rg.state(dim, rank = 1)
sigma = rg.state(dim, rank = 1)

# Get a depolarizing channel
p = 0.5
t_depol = cf.depolarizing(dim, p)

# Apply the channel to the states
rho1 = (t_depol @ rho.reshape(dim**2)).reshape((dim, dim))
sigma1 = (t_depol @ sigma.reshape(dim**2)).reshape((dim, dim))

# The second number should be p times the first
print(tr_dist(rho, sigma))
print(tr_dist(rho1, sigma1))
print(np.allclose(p * tr_dist(rho, sigma), tr_dist(rho1, sigma1)))
```

See more in [the documentation](https://RubenIbarrondo.github.io/quantum_channels/).


