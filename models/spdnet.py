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
    size = max(n, m)
    Q,_ = jnp.linalg.qr(random.uniform(key, shape=(size, size)))
    Q = Q[:n, :m]
    return Q


class BiMapLayer(nn.Module):
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
    """
    MultiMapLayer - projects SPD matrix 
    to multiple different submanifolds
    
    out_dim: dimention of submanifolds
    n_submanifolds: number of submanifolds to project on
    
    learnable parameter: series of orthogonal matrices
    which project SPD matrix to n_submanifolds submanifolds
    """
    out_dim: int
    n_submanifolds: int
    params_init: Callable = multimap_init

    @nn.compact
    def __call__(self, inputs):
        
        def quadratic_form(w, X):
            oper_1 = vmap(lambda w, X: jnp.swapaxes(w, -1, -2) @ X, (None, 0), 0)
            oper_2 = vmap(lambda X, w: X @ w, (0, None), 0)
            return oper_2(oper_1(w, X), w)
        
        submanifold_maps = self.param('Matrix',
                                    self.params_init, # Initialization function for Orthogonal matrix
                                    self.n_submanifolds, inputs.shape[-1], self.out_dim)  # shape info.
        if len(inputs.shape) > 2:
            y = quadratic_form(submanifold_maps, inputs)
        else:
            y = jnp.swapaxes(submanifold_maps, -1, -2) @ inputs @ submanifold_maps
        return y


class MultiBiMapLayer(nn.Module):
    """
    MultiBiMapLayer - projects a series of SPD matricies 
    to their own unique submanifolds
    
    out_dim: dimention of submanifolds
    
    learnable parameter: series of orthogonal matrices
    which project each SPD matrix to their own submanifold
    """
    out_dim: int
    params_init: Callable = multimap_init

    @nn.compact
    def __call__(self, inputs):
        
        def quadratic_form(w, X):
            oper_1 = vmap(lambda w, X: jnp.swapaxes(w, -1, -2) @ X, (-3, -3),-3 )
            oper_2 = vmap(lambda X, w: X @ w, (-3, -3),-3 )
            return oper_2(oper_1(w, X), w)
        
        submanifold_maps = self.param('Matrix',
                                    self.params_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-3], inputs.shape[-1], self.out_dim)  # shape info.
        
        y = quadratic_form(submanifold_maps, inputs)
        
        return y



class ReEigLayer(nn.Module):
    """
    ReEigLayer: prevents from negative eigenvalues
    on projected submanifolds by replacing them with threschold value, 
    also plays role of non-linearity
    """
    threschold: int = 1e-5

    @nn.compact
    def __call__(self, inputs):
        
        def reeig(M):
            evals, evecs = jnp.linalg.eigh(M)
            new_evals = jnp.maximum(evals, self.threschold)
            return evecs @ jnp.diag(new_evals) @ evecs.T
        
        if len(inputs.shape) > 2:
            if len(inputs.shape) > 3:
                y = vmap(vmap(reeig))(inputs)
            else:
                y = vmap(reeig)(inputs)
        else:
            y = reeig(inputs)
        return y
    
    
class LogEigLayer(nn.Module):
    """
    ReEigLayer: last layer before switching from 
    SPD matrix learning to euclidean neural network
    """
    
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
    """
    Triu: converts SPD matrix to a vector
    by taking upper triangular values
    """
    
    @nn.compact
    def __call__(self, inputs):
        
        idx = jnp.triu_indices(inputs.shape[-1])
        if len(inputs.shape) > 2:
            y = vmap(lambda mtx: mtx[idx])(inputs)
        else:
            y = inputs[idx]
        return y


class SPDAvgPooling(nn.Module):
    """
    SPDAvgPooling: for a given set of SPD matricies
    calculates their weighted mean

    optimiser: geometric optimiser to be used for mean calculation
    maxiter: max iteration for computing mean (default=100)
    debug: printing out debug information to trace Nans
    
    learnable parameter: weights for mean
    """
    optimiser: Any
    maxiter: int = 100
    debug: bool = False
    weights_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, inputs):
        weights = self.param('weights', self.weights_init, (inputs.shape[-3], ))  # shape info.
        # weights = jnp.maximum(weights, 1 / (inputs.shape[-3] + 1e-7))
        if len(inputs.shape) > 3:
            def vectorized(inputs):
                return weighted_mean(inputs, weights, self.optimiser, maxiter=self.maxiter, debug=self.debug)
            y = vmap(vectorized)(inputs)
        else:
            y = weighted_mean(inputs, weights, self.optimiser, maxiter=self.maxiter)
        
        return y