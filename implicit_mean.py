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
from jax import vmap, lax, jit

# Optax lib for clipping
import optax

# Distance function
from geomjax.manifolds.utils import pairwise_distance
# Parallelistaion
from geomjax.utils import calc_n_jobs, parallelize_array, merge_parallel_results
# Plotting
import matplotlib.pyplot as plt




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

    # Set randomness
    key,_ = random.split(random.PRNGKey(0))
    
    # Choose a subset
    if X_set.shape[0] > 5:
        X_sample = random.choice(key, X_set, shape=(5,))
    else:
        X_sample = X_set
        
    # Find pairwise distances for each point in a subset
    distances = vmap(pairwise_distance, (0, None, None), 0)(X_sample, X_set, optimiser.manifold.distance)
    # Init mean with a point closest to other points
    if len(X_set.shape) > 2:
        Y = X_set[jnp.argmin(distances)] + jnp.abs(random.uniform(key, shape=(X_set.shape[-1],)) * 1e-4)
    else:
        Y = X_set[jnp.argmin(distances)] + 1e-4

    optim_state = optimiser.init(Y)

    for i in range(maxiter):

        # calculate loss
        if plot_loss_flag or debug:
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

# === Functions for mean derivative ===
# Multiplication
@jit
def grad_multiply_inverse(dyy, d_mixed):
        return -jnp.linalg.solve(dyy, d_mixed)


def weighted_mean_implicit_derivative(x, X, w, manifold):
    """
    Computation of the derivative for the weighted mean
    """

    dy = grad(pairwise_distance, argnums=0)
    dy_projected = lambda x, X, w: manifold.project(x, dy(x, X, manifold.distance, w))  # новая строчка (проекция)

    d2yy = jnp.squeeze(jacobian(dy_projected, argnums=0)(x, X, w))
    # d2yy = manifold.project(x, jnp.squeeze(jacobian(dy_projected, argnums=0)(x, X, w)))
    # d2yy_inv = jnp.linalg.inv(d2yy) # replace inv with solve and move to grad_multiply

    d2xy = jnp.squeeze(jacobian(dy_projected, argnums=1)(x, X, w))  # (2, 22, 2)
    d2wy = jnp.squeeze(jacobian(dy_projected, argnums=2)(x, X, w))  # (2, 22)

    grad_multiply_inverse_batch = vmap(grad_multiply_inverse, (None, 1), 1)

    dfdx = grad_multiply_inverse_batch(d2yy, d2xy) # (2, 22, 2)
    dfdw = grad_multiply_inverse_batch(d2yy, d2wy)

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
    # d2yy_inv = jnp.linalg.inv(d2yy)

    #print(f"d2yy {d2yy} and inv {d2yy_inv}")

    d2xy = jnp.squeeze(jacobian(dy_projected_trace, argnums=1)(x, X, w))  # (10, 5, 5)
    d2wy = jnp.squeeze(jacobian(dy_projected, argnums=2)(x, X, w)).swapaxes(-1, 0)  # (10, 5, 5)
    
    # Vectorize
    grad_multiply_inverse_batch = vmap(grad_multiply_inverse, (None, 0), 0)
    
    # n_cpu = cpu_count()
    dfdx = grad_multiply_inverse_batch(d2yy, d2xy)
    dfdw = grad_multiply_inverse_batch(d2yy, d2wy)

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
        grad_X, grad_w = weighted_mean_implicit_matrix_derivative(x, X, w, manifold)
        out_X = grad_X @ g
        out_w = jnp.trace(grad_w @ g, axis1=-2, axis2=-1)
        return (out_X, out_w)
    else:
        grad_X, grad_w = weighted_mean_implicit_derivative(x, X, w, manifold)
        out_X = grad_X.T @ g
        out_w = grad_w.T @ g
        return (out_X.T, out_w)


weighted_mean.defvjp(weighted_mean_fwd, weighted_mean_bwd)
