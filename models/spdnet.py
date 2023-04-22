# Base of math operations and derivatives
from jax import numpy as jnp
# for initialisation
from jax import random
# for vectorization
from jax import vmap

# Defining data for class
from typing import Any, Callable, Sequence

# Flax framework for Deep Learning
from flax.core import freeze, unfreeze
from flax import linen as nn

# Implicit manifold mean
from geomjax.implicit_mean import weighted_mean


def bimap_init(key, n, m):
    Q,_ = jnp.linalg.qr(random.uniform(key, shape=(n, n)))
    Q = Q[:m, :]
    return Q.T


class BiMapLayer(nn.Module):
    out_dim: int
    matrix_init: Callable = bimap_init

    @nn.compact
    def __call__(self, inputs):
        
        def quadratic_form(w, X):
            oper_1 = vmap(lambda w, X: w.T @ X, (None, 0), 0)
            oper_2 = vmap(lambda X, w: X @ w, (0, None), 0)
            return oper_2(oper_1(w, X), w)
        
        gen_key = random.PRNGKey(0)
        mapping_matrix = self.param('Matrix',
                                    self.matrix_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-1], self.out_dim)  # shape info.
        if len(inputs.shape) > 2:
            y = quadratic_form(mapping_matrix, inputs)
        else:
            y = mapping_matrix.T @ inputs @ mapping_matrix
        return y


def multimap_init(key, n_submanifolds, n, m):
    params = []
    for _ in range(n_submanifolds):
        key, _ = random.split(key, 2)
        params.append(bimap_init(key, n, m))
    return jnp.array(params)


class MultiMapLayer(nn.Module):
    out_dim: int
    n_submanifolds: int
    params_init: Callable = multimap_init

    @nn.compact
    def __call__(self, inputs):
        
        def quadratic_form(w, X):
            oper_1 = vmap(lambda w, X: jnp.swapaxes(w, -1, -2) @ X, (None, 0), 0)
            oper_2 = vmap(lambda X, w: X @ w, (0, None), 0)
            return oper_2(oper_1(w, X), w)
        
        gen_key = random.PRNGKey(0)
        submanifold_maps = self.param('Matrix',
                                    self.params_init, # Initialization function for Orthogonal matrix
                                    self.n_submanifolds, inputs.shape[-1], self.out_dim)  # shape info.
        if len(inputs.shape) > 2:
            y = quadratic_form(submanifold_maps, inputs)
        else:
            y = jnp.swapaxes(submanifold_maps, -1, -2) @ inputs @ submanifold_maps
        return y


class ReEigLayer(nn.Module):
    threschold: int

    @nn.compact
    def __call__(self, inputs):
        def reeig(M):
            evals, evecs = jnp.linalg.eigh(M)
            evals = jnp.maximum(evals, self.threschold)
            return evecs @ jnp.diag(evals) @ evecs.T
        
        if len(inputs.shape) > 2:
            if len(inputs.shape) > 3:
                y = vmap(vmap(reeig))(inputs)
            else:
                y = vmap(reeig)(inputs)
        else:
            y = reeig(inputs)
        return y
    
    
class LogEigLayer(nn.Module):
    
    @nn.compact
    def __call__(self, inputs):
        def logeig(M):
            evals, evecs = jnp.linalg.eigh(M)
            evals = jnp.log(evals)
            return evecs @ jnp.diag(evals) @ evecs.T
        
        if len(inputs.shape) > 2:
            y = vmap(logeig)(inputs)
        else:
            y = logeig(inputs)
        return y
    
    
class Triu(nn.Module):
    
    @nn.compact
    def __call__(self, inputs):
        
        idx = jnp.triu_indices(inputs.shape[-1])
        if len(inputs.shape) > 2:
            y = vmap(lambda mtx: mtx[idx])(inputs)
        else:
            y = inputs[idx]
        return y


class SPDAvgPooling(nn.Module):
    optimiser: Any
    maxiter: int = 100
    weights_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, inputs):
        weights = self.param('weights', self.weights_init, (inputs.shape[-3], ))  # shape info.
        if len(inputs.shape) > 3:
            def vectorized(inputs):
                return weighted_mean(inputs, weights, self.optimiser, maxiter=self.maxiter)
            y = vmap(vectorized)(inputs)
        else:
            y = weighted_mean(inputs, weights, self.optimiser, maxiter=self.maxiter)
        
        return y