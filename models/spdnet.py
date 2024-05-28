# Base of math operations and derivatives
from jax import numpy as jnp
# for initialisation
from jax import random
# for vectorization
from jax import vmap, jit

# Defining data for class
from typing import Any, Callable

# Flax framework for Deep Learning
from flax import linen as nn

# Implicit manifold mean
from geomjax.implicit_mean import weighted_mean


# === Functions for bimap ====
# Initializer
def bimap_init(key, n, m):
    size = max(n, m)
    Q,_ = jnp.linalg.qr(random.uniform(key, shape=(size, size)))
    Q = Q[:n, :m]
    return Q

# Multiplication
@jit
def bimap_quadratic_form(w, X):
    oper_1 = vmap(lambda w, X: w.T @ X, (None, 0), 0)
    oper_2 = vmap(lambda X, w: X @ w, (0, None), 0)
    return oper_2(oper_1(w, X), w)

# Layer itself
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
        
        # gen_key = random.PRNGKey(0)
        mapping_matrix = self.param('Matrix',
                                    self.matrix_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-1], self.out_dim)  # shape info.
        if len(inputs.shape) > 2:
            y = bimap_quadratic_form(mapping_matrix, inputs)
        else:
            y = mapping_matrix.T @ inputs @ mapping_matrix
        return y


# === Functions for MultiMapLayer ====
# Initializer
def multimap_init(key, n_submanifolds, n, m):
    params = []
    for _ in range(n_submanifolds):
        key, _ = random.split(key, 2)
        params.append(bimap_init(key, n, m))
    return jnp.array(params)

# Multiplication
@jit
def multimap_quadratic_form(w, X):
    oper_1 = vmap(lambda w, X: jnp.swapaxes(w, -1, -2) @ X, (None, 0), 0)
    oper_2 = vmap(lambda X, w: X @ w, (0, None), 0)
    return oper_2(oper_1(w, X), w)

# Layer itself
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
        
        submanifold_maps = self.param('Matrix',
                                    self.params_init, # Initialization function for Orthogonal matrix
                                    self.n_submanifolds, inputs.shape[-1], self.out_dim)  # shape info.
        if len(inputs.shape) > 2:
            y = multimap_quadratic_form(submanifold_maps, inputs)
        else:
            y = jnp.swapaxes(submanifold_maps, -1, -2) @ inputs @ submanifold_maps
        return y

# === Functions for MultiBiMapLayer ===
# Multiplication
@jit
def multibimap_quadratic_form(w, X):
    oper_1 = vmap(lambda w, X: jnp.swapaxes(w, -1, -2) @ X, (-3, -3),-3 )
    oper_2 = vmap(lambda X, w: X @ w, (-3, -3),-3 )
    return oper_2(oper_1(w, X), w)

# Layer itself
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
        
        submanifold_maps = self.param('Matrix',
                                    self.params_init, # Initialization function for Orthogonal matrix
                                    inputs.shape[-3], inputs.shape[-1], self.out_dim)  # shape info.
        
        y = multibimap_quadratic_form(submanifold_maps, inputs)
        
        return y

# === Functions for ReEig ===
# Algorithm
@jit
def reeig(M, threschold):
    evals, evecs = jnp.linalg.eigh(M)
    new_evals = jnp.maximum(evals, threschold)
    return evecs @ jnp.diag(new_evals) @ evecs.T

# Layer itself
class ReEigLayer(nn.Module):
    """
    ReEigLayer: prevents from negative eigenvalues
    on projected submanifolds by replacing them with threschold value, 
    also plays role of non-linearity
    """
    threschold: int = 1e-5

    @nn.compact
    def __call__(self, inputs):
        
        if len(inputs.shape) > 2:
            if len(inputs.shape) > 3:
                y = vmap(vmap(reeig, (0, None)), (0, None))(inputs, self.threschold)
            else:
                y = vmap(reeig, (0, None))(inputs, self.threschold)
        else:
            y = reeig(inputs, self.threschold)
        return y
    
# === Functions for LogEig ===  
# Algorithm
@jit
def logeig(M):
    evals, evecs = jnp.linalg.eigh(M)
    evals = jnp.log(evals)
    return evecs @ jnp.diag(evals) @ evecs.T

# Layer itself
class LogEigLayer(nn.Module):
    """
    ReEigLayer: last layer before switching from 
    SPD matrix learning to euclidean neural network
    """
    
    @nn.compact
    def __call__(self, inputs):
        
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

def mean_weights_init(key, shape):
    w = random.uniform(key, shape=shape) * 1e-3 + 1.0
    return w / jnp.linalg.norm(w)

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
    plot_loss_flag: bool = False
    weights_init: Callable = mean_weights_init #nn.initializers.uniform()

    @nn.compact
    def __call__(self, inputs):
        weights = self.param('weights', self.weights_init, (inputs.shape[-3], )) 
        if len(inputs.shape) > 3:
            def vectorized(inputs):
                return weighted_mean(inputs, weights, self.optimiser, maxiter=self.maxiter, debug=self.debug)
            y = vmap(vectorized)(inputs)
        else:
            y = weighted_mean(inputs, weights, self.optimiser, plot_loss_flag=self.plot_loss_flag, maxiter=self.maxiter, debug=self.debug)
        
        return y
    


# For future attempts to make uuniversal bimap

# # Projection initializer
# def generate_stiefel(key, n, m):
#     size = max(n, m)
#     Q,_ = jnp.linalg.qr(random.uniform(key, shape=(size, size)))
#     Q = Q[:n, :m]
#     return Q

# # Layer initializer
# def bimap_init(key, n, m, n_in, n_out):
#     # generate submanifolds for one input
#     gen_submanifold = vmap(lambda i: generate_stiefel(key, n, m))
#     # now apply to all inputs
#     gen_param = vmap(lambda i: gen_submanifold(range(n_out)))(range(n_in))
#     return gen_param

# # Multiplication
# @jit
# def bimap_quadratic_form(W, X):
#     res_1 = jnp.einsum('akn,abnm->abkm', X, W)
#     res_2 = jnp.einsum('abnm,abnk->abmk', W, res_1).squeeze()
#     return res_2

# # Layer itself
# class BiMapLayer(nn.Module): # get n matrices, out n*m, where m submanifolds
#     """
#     BiMapLayer - projects SPD matrix 
#     to one submanifold
    
#     n_manifolds_in: number of matrices in input, which will be projected
#     out_dim: dimention of submanifold
#     n_submanifolds: nmber of submanifolds for projection per each input matrix
    
#     learnable parameter: orthogonal matrix which projects 
#     SPD matrix to it's own submanifold
#     """
#     out_dim: int
#     n_manifolds_in: int = 1
#     n_submanifolds: int = 1

#     matrix_init: Callable = bimap_init

#     @nn.compact
#     def __call__(self, inputs):
        
#         mapping_matrix = self.param('Matrix',
#                                     self.matrix_init, # Initialization function for Orthogonal matrix
#                                     inputs.shape[-1], # n
#                                     self.out_dim, # m
#                                     self.n_manifolds_in, # n_in
#                                     self.n_submanifolds # n_out
#                                     )  
#         if inputs.ndim == 2:
#             y = bimap_quadratic_form(mapping_matrix, inputs[None, :])
#         elif self.n_manifolds_in != inputs.shape[0]:
#             y = vmap(lambda x: bimap_quadratic_form(mapping_matrix, x))(inputs)
#         else:
#             y = bimap_quadratic_form(mapping_matrix, inputs)
#         return y