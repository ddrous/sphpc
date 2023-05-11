import os
import jax
import jax.numpy as jnp


__version__ = "0.1.0"       ## Package version  ## TODO check if okay to do this here

PREALLOCATE = False
if not PREALLOCATE:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"       ## Preallocate 90% of memory


FLOAT64 = False
jax.config.update("jax_enable_x64", FLOAT64)   ## Use double precision by default

jnp.set_printoptions(linewidth=jnp.inf)         ## Print arrays on the same line
