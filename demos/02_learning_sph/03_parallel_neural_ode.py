"""
					Neural ODEs in Parallel
   
"""


##

#### Flatten paramters using Kdger's Equinox: see https://github.com/google/flax/issues/5#issuecomment-595676137

#%%

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sphpc import flatten_params, unflatten_params


class MLP(nn.Module):
  features: int
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.features)(x)
    x = nn.relu(x)
    x = nn.Dense(self.features)(x)
    x = nn.relu(x)
    x = nn.Dense(self.features)(x)
    return x

model = MLP(features=10)

key = jax.random.PRNGKey(0)
init_data = jax.numpy.zeros((10, 3))

params = model.init(key, init_data)


flat_params, flat_shapes, tree_def = flatten_params(params)
new_params = unflatten_params(flat_params, flat_shapes, tree_def)


# Compare the old and new params
np.allclose(params['params']['Dense_1']['bias'], new_params['params']['Dense_1']['bias'])


# %%
