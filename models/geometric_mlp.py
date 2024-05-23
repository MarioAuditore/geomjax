# Base of math operations and derivatives
from jax import numpy as jnp
# for initialisation
from jax import random
# for vectorization
from jax import vmap

# Defining data for class
from typing import Callable

# Flax framework for Deep Learning
from flax import linen as nn

# Weights initialization
from geomjax.models.spdnet import bimap_init


# === Functions for Stiefel MLP ====

# Layer itself
class StiefelLinear(nn.Module):
    """
    StiefelLinear - MLP with weight matrix on Stiefel manifold
    
    out_dim: dimention of output
    
    learnable parameter: orthogonal matrix dim_in x dim_out
    """
    out_dim: int
    bias : bool = False
    matrix_init: Callable = bimap_init

    @nn.compact
    def __call__(self, inputs):
        
        if self.bias:
            mapping_matrix = self.param('Matrix',
                                        self.matrix_init, # Initialization function for Orthogonal matrix
                                        inputs.shape[-1] + 1, self.out_dim)  # shape info.
            if len(inputs.shape) < 2:
                inputs = jnp.expand_dims(inputs, 0)
                ones  = jnp.expand_dims(jnp.ones(inputs.shape[0]), 0)
            else:
                ones  = jnp.expand_dims(jnp.ones(inputs.shape[0]), 0).T
            
            x = jnp.hstack([inputs, ones])
            y = x @ mapping_matrix

        else:
            mapping_matrix = self.param('Matrix',
                                        self.matrix_init, # Initialization function for Orthogonal matrix
                                        inputs.shape[-1], self.out_dim)  # shape info.
            y = inputs @ mapping_matrix
        
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
    SphereLinear - MLP with weight matrix on Hypersphere
    
    out_dim: dimention of output
    
    learnable parameter: matrix with normalised basis
    """
    out_dim: int
    bias : bool = False
    matrix_init: Callable = sphere_init

    @nn.compact
    def __call__(self, inputs):
        
        if self.bias:
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

        else:
            mapping_matrix = self.param('Matrix',
                                        self.matrix_init, # Initialization function for Orthogonal matrix
                                        inputs.shape[-1], self.out_dim)  # shape info.
            y = inputs @ mapping_matrix.T
        
        return y
