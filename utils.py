# Base of math operations and derivatives
from jax import numpy as jnp
# For batch functions
from jax import vmap, jit
# Parallelisation
from jax import device_count


# === functions for parallelism ===
# Currently not used for stability

def calc_n_jobs(n_samples):
    n_cores = device_count()
    
    while n_cores > 1:
        if n_samples % n_cores == 0:
            return n_cores
        else:
            n_cores -= 1
    return n_cores


def parallelize_array(X, n_jobs):
    return X.reshape(n_jobs, X.shape[0] // n_jobs, *X.shape[1:])

def merge_parallel_results(result):
    return result.reshape(result.shape[0] * result.shape[1], *result.shape[2:])