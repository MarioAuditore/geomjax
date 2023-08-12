# Base of math operations and derivatives
from jax import numpy as jnp
# for initialisation
from jax import random
# for vectorization
from jax import vmap, jit
# Parallelistaion
from jax import pmap


# Defining data for class
from typing import Any, Callable, Sequence

# Flax framework for Deep Learning
from flax.core import freeze, unfreeze
from flax import linen as nn

# Weights initialization
from geomjax.models.spdnet import bimap_init


# === Functions for Stiefel MLP ====

# Layer itself
class StiefelLinear(nn.Module):
    """
    BiMapLayer - projects SPD matrix 
    to one submanifold
    
    out_dim: dimention of submanifold
    
    learnable parameter: orthogonal matrix which projects 
    SPD matrix to it's own submanifold
    """
    out_dim: int
    matrix_init: Callable = bimap_init

    @nn.compact
    def __call__(self, inputs):
        
        # gen_key = random.PRNGKey(0)
        mapping_matrix = self.param('Matrix',
                                    self.matrix_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-1] + 1, self.out_dim)  # shape info.
        x = jnp.vstack([jnp.ones(inputs.shape[0]), inputs])
        y = x @ mapping_matrix.T
        return y


# === Functions for Sphere MLP ====
# Initializer
def sphere_init(key, n, m):
    W = random.uniform(key, shape=(n, m), minval=-1.0, maxval=1.0).T
    W = vmap(lambda x : x / jnp.linalg.norm(x))(W)
    return W

# Layer itself
class SphereLinear(nn.Module):
    """
    BiMapLayer - projects SPD matrix 
    to one submanifold
    
    out_dim: dimention of submanifold
    
    learnable parameter: orthogonal matrix which projects 
    SPD matrix to it's own submanifold
    """
    out_dim: int
    matrix_init: Callable = sphere_init

    @nn.compact
    def __call__(self, inputs):
        
        # gen_key = random.PRNGKey(0)
        mapping_matrix = self.param('Matrix',
                                    self.matrix_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-1] + 1, self.out_dim)  # shape info.
        if len(inputs.shape) < 2:
            inputs = jnp.expand_dims(inputs, 0)
            ones  = jnp.expand_dims(jnp.ones(inputs.shape[0]), 0)
        else:
            ones  = jnp.expand_dims(jnp.ones(inputs.shape[0]), 0).T
        
        x = jnp.hstack([inputs, ones])
        y = x @ mapping_matrix.T
        return y
