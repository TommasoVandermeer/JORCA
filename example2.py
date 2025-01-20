import jax.numpy as jnp


NAN_ORCA_LINE = jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])

# Tile the NAN_ORCA_LINE for a number of times in dimension 0
num_tiles = 3
tiled_nan_orca_line = jnp.tile(NAN_ORCA_LINE, (num_tiles, 1, 1))

print(tiled_nan_orca_line)
print(jnp.empty((1,2,2)))