import jax.numpy as jnp
from jax import jit, vmap, lax, debug, random

# TODO: Add generate random humans parameters function
# TODO: Add generate random circular crossing scenario initial conditions function
# TODO: Add generate random parallel traffica initial conditions function

def get_standard_humans_parameters(n_humans:int, dt:float):
    """
    Returns the standard parameters of the ORCA for the humans in the simulation. Parameters are the same for all humans in the form:
    (radius, time_horizon, v_max, safety_space).

    args:
    - n_humans: int - Number of humans in the simulation.

    outputs:
    - parameters (n_humans, 4) - Standard parameters for the humans in the simulation.
    """
    single_params = jnp.array([0.3, dt, 1., 0.])
    return jnp.tile(single_params, (n_humans, 1))