"""
============================================================
--- Gradient descent based mean with implicit derivative ---
============================================================
"""
# Base of math operations and derivatives
from jax import numpy as jnp
# Custom derivative declaration
from jax import grad, jacobian, custom_vjp
# Random generator
from jax import random
# Jax functions for jit
from functools import partial
# Vectorization
from jax import vmap, lax
# Parallelisation
from jax import pmap
# Optax lib for clipping
import optax

import numpy as np

# Distance function
from geomjax.manifolds.utils import pairwise_distance
# Parallelistaion
from geomjax.utils import calc_n_jobs, parallelize_array, merge_parallel_results
# Plotting
import matplotlib.pyplot as plt


# @implicit_diff.custom_root(grad(pairwise_distance))
def gradient_descend_weighted_mean(X_set, weights, optimiser, plot_loss_flag, maxiter, debug = False):
    """
    Weighted mean calculation as an optimization problem:
    find point, which minimises pairwise distances in the given set of points
    """

    # if no weights are provided, they are the same
    if weights == None:
        weights = jnp.ones(X_set.shape[0]) / X_set.shape[0]
        

    # array to plot loss
    if plot_loss_flag:
        plot_loss = []

    key,_ = random.split(random.PRNGKey(0))
    # init mean with random element from set and move it a bit
    if len(X_set.shape) > 2:
        Y = X_set[np.random.randint(0, X_set.shape[0], (1,))][0] + jnp.abs(random.uniform(key, shape=(X_set.shape[-1],)) * 1e-4)
    else:
        Y = X_set[np.random.randint(0, X_set.shape[0], (1,))][0] + 1e-4

    optim_state = optimiser.init(Y)
    
    average_grad = None

    for i in range(maxiter):

        # calculate loss
        loss = pairwise_distance(Y, X_set, optimiser.manifold.distance, weights)

        # compute gradient
        euclid_grad = grad(pairwise_distance, argnums=0)(Y, X_set, optimiser.manifold.distance, weights)

        if debug:
            print(f"Iter {i} | loss = {loss}")
            print(f"Euclid grad norm:{jnp.linalg.norm(euclid_grad)} | grad / param: {jnp.linalg.norm(euclid_grad) / jnp.linalg.norm(Y)}")

        # Gradient clipping
        if i == 0:
            # Init gradient clipping on first iteration
            grad_clipper = optax.adaptive_grad_clip(jnp.linalg.norm(euclid_grad) / jnp.linalg.norm(Y))
            clipper_state = grad_clipper.init(Y)
        else:
            # apply clipping
            euclid_grad, clipper_state = grad_clipper.update(euclid_grad, clipper_state, Y)

        Y, optim_state = optimiser.update(Y, euclid_grad, optim_state)

        if debug:
            print(f"Updated mean: {jnp.linalg.eigh(Y)[0]}")

        if plot_loss_flag:
            # collect loss for plotting
            plot_loss.append(loss)

    if plot_loss_flag:
        print(f"Total loss: {pairwise_distance(Y, X_set, optimiser.manifold.distance, weights)}")
        fig, ax = plt.subplots()
        ax.plot(plot_loss)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.show()

    return Y



def weighted_mean_implicit_derivative(x, X, w, manifold):
    """
    Computation of the derivative for the weighted mean
    """

    dy = grad(pairwise_distance, argnums=0)
    dy_projected = lambda x, X, w: manifold.project(x, dy(x, X, manifold.distance, w))  # новая строчка (проекция)

    d2yy = jnp.squeeze(jacobian(dy_projected, argnums=0)(x, X, w))
    # d2yy = manifold.project(x, jnp.squeeze(jacobian(dy_projected, argnums=0)(x, X, w)))
    d2yy_inv = jnp.linalg.inv(d2yy)

    d2xy = jnp.squeeze(jacobian(dy_projected, argnums=1)(x, X, w))  # (2, 22, 2)
    d2wy = jnp.squeeze(jacobian(dy_projected, argnums=2)(x, X, w))  # (2, 22)

    def grad_multiply_inverse(dyy_inv, d_mixed):
        return -dyy_inv @ d_mixed

    grad_multiply_inverse_batch = vmap(grad_multiply_inverse, (None, 1), 1)

    dfdx = grad_multiply_inverse_batch(d2yy_inv, d2xy) # (2, 22, 2)
    dfdw = grad_multiply_inverse_batch(d2yy_inv, d2wy)

    return dfdx, dfdw


def weighted_mean_implicit_matrix_derivative(x, X, w, manifold):
    """
    Computation of the derivative for the weighted mean
    """

    dy = grad(pairwise_distance, argnums=0)
    dy_projected = lambda x, X, w: manifold.project(x, dy(x, X, manifold.distance, w))  
    dy_projected_trace = lambda x, X, w: jnp.trace(manifold.project(x, dy(x, X, manifold.distance, w)))  

    d2yy = jnp.squeeze(jacobian(dy_projected_trace, argnums=0)(x, X, w))
    # d2yy = manifold.project(x, jnp.squeeze(jacobian(dy_projected_trace, argnums=0)(x, X, w)))
    d2yy_inv = jnp.linalg.inv(d2yy)

    #print(f"d2yy {d2yy} and inv {d2yy_inv}")

    d2xy = jnp.squeeze(jacobian(dy_projected_trace, argnums=1)(x, X, w))  # (10, 5, 5)
    d2wy = jnp.squeeze(jacobian(dy_projected, argnums=2)(x, X, w)).swapaxes(-1, 0)  # (10, 5, 5)
    
    def grad_multiply_inverse(dyy_inv, d_mixed):
        return -dyy_inv @ d_mixed

    # Vectorize
    grad_multiply_inverse_batch = vmap(grad_multiply_inverse, (None, 0), 0)
    # Find suitable for pmap amount of cores
    # n_jobs = calc_n_jobs(X.shape[0])

    # If there is no parallelisation option
    # if n_jobs == 1:
    dfdx = grad_multiply_inverse_batch(d2yy_inv, d2xy)
    dfdw = grad_multiply_inverse_batch(d2yy_inv, d2wy)
    # else:
    #     # print(f"DEBUG: Derivative in parallel mode on {n_jobs} devices")
    #     grad_multiply_inverse_batch_parallel = pmap(grad_multiply_inverse_batch, in_axes=(None, 0), out_axes=0)
    #     # Split by jobs
    #     d2xy_batched = parallelize_array(d2xy, n_jobs)
    #     d2wy_batched = parallelize_array(d2wy, n_jobs)
    #     # Parallel computation
    #     dfdx_batched = grad_multiply_inverse_batch_parallel(d2yy_inv, d2xy_batched)
    #     dfdw_batched = grad_multiply_inverse_batch_parallel(d2yy_inv, d2wy_batched)
    #     # Merge results
    #     dfdx = merge_parallel_results(dfdx_batched)
    #     dfdw = merge_parallel_results(dfdw_batched)

    return dfdx, dfdw



@partial(custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def weighted_mean(X, w, optimiser, plot_loss_flag=False, maxiter=200, debug = False):
    return gradient_descend_weighted_mean(X_set=X, 
                                          weights=w, 
                                          optimiser=optimiser, 
                                          plot_loss_flag=plot_loss_flag,
                                          maxiter=maxiter,
                                          debug=debug)


def weighted_mean_fwd(X, w, optimiser, plot_loss_flag, maxiter, debug):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    x = weighted_mean(X, w, optimiser, plot_loss_flag, maxiter, debug)
    return x, (x, X, w, optimiser.manifold)


def weighted_mean_bwd(optimiser, plot_loss_flag, maxiter, debug, res, g):
    x, X, w, manifold = res  # Gets residuals computed in f_fwd
    if len(x.shape) > 1:
        n_jobs = calc_n_jobs(X.shape[0])
        if n_jobs == 1:
            grad_X, grad_w = weighted_mean_implicit_matrix_derivative(x, X, w, manifold)
        else:
            # print(f"DEBUG: Derivative in parallel mode on {n_jobs} devices")
            # Make parallel versiob
            weighted_mean_implicit_matrix_derivative_parallel = pmap(
                lambda X, w: weighted_mean_implicit_matrix_derivative(x, X, w, manifold),
                in_axes=(0, 0), 
                out_axes=0)
            # Split arrays by devices
            X_batched = parallelize_array(X, n_jobs)
            w_batched = parallelize_array(w, n_jobs)
            # Compute
            grad_X_batched, grad_w_batched = weighted_mean_implicit_matrix_derivative_parallel(X_batched, w_batched)
            # Merge results
            grad_X = merge_parallel_results(grad_X_batched)
            grad_w = merge_parallel_results(grad_w_batched)

        out_X = grad_X @ g
        out_w = jnp.trace(grad_w @ g, axis1=-2, axis2=-1)
        return (out_X, out_w)
    else:
        grad_X, grad_w = weighted_mean_implicit_derivative(x, X, w, manifold)
        out_X = grad_X.T @ g
        out_w = grad_w.T @ g
        return (out_X.T, out_w)


weighted_mean.defvjp(weighted_mean_fwd, weighted_mean_bwd)
