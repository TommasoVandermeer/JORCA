import jax.numpy as jnp
from jax import jit, lax

NAN_ORCA_LINE = jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])

@jit
def test_nan(arr:jnp.array):
    @jit
    def _fori_body(i, val):
        inp, out = val
        out = lax.cond(
            jnp.any(jnp.isnan(inp[i])),
            lambda _: out.at[i].set(True),
            lambda _: out,
            operand=None
        )
        return inp, out
    arr, out = lax.fori_loop(0, arr.shape[0], _fori_body, (arr, jnp.zeros(arr.shape[0], dtype=bool)))
    return out

# Tile the NAN_ORCA_LINE for a number of times in dimension 0
num_tiles = 3
tiled_nan_orca_line = jnp.tile(NAN_ORCA_LINE, (num_tiles, 1, 1))
tiled_nan_orca_line = tiled_nan_orca_line.at[-1].set(jnp.array([[1., 1.], [1., 1.]]))

print(tiled_nan_orca_line)
print(test_nan(tiled_nan_orca_line))

@jit
def compute_neighbors(position:jnp.ndarray, other_humans_positions:jnp.ndarray):
    relative_positions = other_humans_positions - position
    squared_distances = jnp.sum(relative_positions * relative_positions, axis=1)
    max_distance = 3.
    neighbors = lax.fori_loop(
        0,
        len(squared_distances),
        lambda i, val: lax.cond(
            squared_distances[i] < max_distance**2,
            lambda _: val.at[i].set(True),
            lambda _: val,
            operand=None
        ),
        jnp.zeros(len(squared_distances), dtype=bool)
    )
    return neighbors

position = jnp.array([0., 0.])
other_humans_positions = jnp.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.], [6., 6.], [7., 7.], [8., 8.], [9., 9.], [10., 10.]])

print(compute_neighbors(position, other_humans_positions))