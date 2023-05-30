
from functools import lru_cache, partial
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

## parametrise with period = 2*pi
def periodic_dist(x, y):
  L = 2 * np.pi * np.array([-1, 0, 1])
  z1 = x[0] - y[0] - L
  z2 = x[1] - y[1] - L
  z3 = x[2] - y[2] - L

  vects = np.array(np.meshgrid(z1, z2, z3, indexing='ij')).T.reshape(-1,3)

  return np.min(np.linalg.norm(vects, axis=-1))


def kd_tree_neighbors_search(positions, query_positions, smoothing_length):
    """ Find the neighbours of each particle """
    from sklearn import neighbors

    # neighbor_ids, distances = neighbors.KDTree(
    #     positions,
    # ).query_radius(
    #     query_positions,
    #     smoothing_length,
    #     return_distance=True,
    #     sort_results=True
    # )

    neighbor_ids, distances = neighbors.BallTree(
        positions, metric=periodic_dist
    ).query_radius(
        query_positions,
        smoothing_length,
        return_distance=True,
        sort_results=True
    )

    # Drop the element itself (if using KDTree)
    neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
    distances = [ np.delete(x, 0) for x in distances]

    return neighbor_ids, distances





Cell = namedtuple('Cell', ['id', 'neighbors'])

def compute_id(i, j, k, n_cells):
    return i + j*n_cells + k*n_cells*n_cells

# @lru_cache(maxsize=None)
def construct_cells_for_nn_search(d_lim, h):
  n = int(np.ceil(d_lim/h))

  cells = {}
  for k in range(n):
    for j in range(n):
      for i in range(n):
        cell_id = compute_id(i,j,k, n)

        neighbors = []
        for k_ in (k-1, k, k+1):
          for j_ in (j-1, j, j+1):
            for i_ in (i-1, i, i+1):
              neighbors.append(compute_id(i_%n, j_%n, k_%n, n))

        cells[cell_id] = Cell(cell_id, neighbors)

  return cells



## This can be easily vectorised
def find_cell_for_point(x, h, n):
  return compute_id(int(x[0]/h), int(x[1]/h), int(x[2]/h), n)

"""
@Brief: This function computes the neighbors of each particle in a periodic domain
@Input: X: (N, 3) array of  particles positions at wich we query
@d_lim: period of the cubic domain
@h: query radius
"""
def periodic_fixed_radius_nearest_neighbor(X, d_lim, h, cells):

  N = X.shape[0]                        ## number of particles
  n = int(np.ceil(d_lim/h))             ## number of grid cells in each direction

  points_to_cells = []
  for i in range(N):
    points_to_cells.append(find_cell_for_point(X[i], h, n))
  points_to_cells = np.array(points_to_cells)

  cells_to_points = {}
  for cell in cells.values():
    cells_to_points[cell.id] = np.argwhere(points_to_cells==cell.id)[:,0]

  neighbor_points = []
  neighbor_dists = []

  for i in range(N):
    neighbor_cells = cells[points_to_cells[i]].neighbors

    nei_ids = []
    nei_dists = []
    for cell_id in neighbor_cells:

      for j in cells_to_points[cell_id]:
        d = distance(X[i], X[j], d_lim)
        if d <= h and j != i:
          nei_ids.append(j)
          nei_dists.append(d)

    neighbor_points.append(nei_ids)
    neighbor_dists.append(nei_dists)

  return neighbor_points, neighbor_dists




def periodic_fixed_radius_nearest_neighbor(X, d_lim, h, cells):

  N = X.shape[0]               ## number of particles
  n = int(np.ceil(d_lim/h))             ## number of grid cells in each direction

  ## This can be easily vectorised
  def find_cell(x):
    return compute_id(int(x[0]/h), int(x[1]/h), int(x[2]/h), n)
  # find_cell_vec = np.vectorize(find_cell) ## TODO: vectorise this

  # cells = construct_cells(n)

  points_to_cells = []
  for i in range(N):
    points_to_cells.append(find_cell(X[i]))
  points_to_cells = np.array(points_to_cells)

  def cell_to_points_func(cell_id, points_to_cells):
    return list(np.argwhere(points_to_cells==cell_id)[:,0])

  cells_to_points = {}
  for cell in cells.values():
    cells_to_points[cell.id] = cell_to_points_func(cell.id, points_to_cells)


  def distance(x, y):
    diff1 = np.abs(x[0] - y[0])
    diff2 = np.abs(x[1] - y[1])
    diff3 = np.abs(x[2] - y[2])
    return np.sqrt(np.min([diff1, d_lim-diff1])**2 +\
					np.min([diff2, d_lim-diff2])**2  +\
					np.min([diff3, d_lim-diff3])**2	)

  neighbor_points = []
  neighbor_dists = []

  for i in range(N):
    neighbor_cells = cells[points_to_cells[i]].neighbors

    nei_ids = []      ## TODO Don't track gradients here
    nei_dists = []
    for cell_id in neighbor_cells:

      ## Avoid the below with vectorisation  ## TODO
      for j in cells_to_points[cell_id]:
        d = distance(X[i], X[j])
        if d <= h and j != i:
          nei_ids.append(j)
          nei_dists.append(d)

    neighbor_points.append(nei_ids)
    neighbor_dists.append(nei_dists)

  return neighbor_points, neighbor_dists



# @partial(jax.jit, static_argnames=('n',))
# @lru_cache(maxsize=None)
def construct_cells_for_nn_search_jax(d_lim, h):

  n = int((d_lim / h) + ((d_lim % h) != 0))
  cells = np.zeros((n*n*n, 27), dtype=jnp.int32)
  # cells = jax.lax.stop_gradient(cells)

  for k in range(n):
    for j in range(n):
      for i in range(n):
        cell_id = compute_id(i,j,k, n)

        neighbors = []
        for k_ in (k-1, k, k+1):
          for j_ in (j-1, j, j+1):
            for i_ in (i-1, i, i+1):
              neighbors.append(compute_id(i_%n, j_%n, k_%n, n))

        cells[cell_id] = jnp.array(neighbors)
        # cells.at[cell_id].set(jnp.array(neighbors))

  return cells






## This can be easily vectorised
def find_cell(x, h, n):
  return jnp.array(compute_id(x[0]/h, x[1]/h, x[2]/h, n), dtype=int)
find_cell_vec = jax.vmap(find_cell, in_axes=(0, None, None), out_axes=0)


def distance(x, y, d_lim):   ## TODO: tensorise this
  diff = jnp.abs(x-y)
  diff_min = jnp.min(jnp.vstack((diff, d_lim-diff)).T, axis=-1)
  return jnp.sqrt(jnp.sum(diff_min**2, axis=0))
distance_vec = jax.vmap(distance, in_axes=(None,0, None), out_axes=0)


# @partial(jax.jit, static_argnames=('h', 'X', 'cells_to_points'))
# @jax.jit
def find_neighbours_of_i_in_cell(i, cell_id, X, cells_to_points, h):
  neigh_ids = cells_to_points[cell_id]
  neigh_ids = neigh_ids[(neigh_ids > -1) & (neigh_ids != i)] ## remove the -1s (see argwhere) and the self

  neigh_dists = distance_vec(X[i], X[neigh_ids])

  return neigh_ids[neigh_dists <= h], neigh_dists[neigh_dists <= h]


def find_points_in_cell(cell_id, points_to_cells):
  return jnp.argwhere(points_to_cells==cell_id, size=points_to_cells.shape[0], fill_value=-1)[:,0]

find_points_in_cell_vec = jax.vmap(find_points_in_cell, in_axes=(0, None), out_axes=0)

def find_points_in_cell_count(cell_id, points_to_cells):
  return jnp.sum(points_to_cells==cell_id)

find_points_in_cell_count_vec = jax.vmap(find_points_in_cell_count, in_axes=(0,None), out_axes=0)


# def distances_from_point_i_to_cell(i, cell_id, X, cells_to_points, d_lim, h):
#     neigh_ids = cells_to_points[cell_id, :]

#     neigh_ids = neigh_ids[(neigh_ids > -1) & (neigh_ids != i)] ## remove the -1s (see argwhere above) and the self
#     # neigh_ids = neigh_ids[neigh_ids != i]
#     neigh_dists = distance_vec(X[i], X[neigh_ids], d_lim)

#     return neigh_ids[neigh_dists <= h], neigh_dists[neigh_dists <= h]

# distances_from_point_i_to_cell_vec = jax.vmap(distances_from_point_i_to_cell, in_axes=(None, 0, None, None, None, None), out_axes=0)


def periodic_fixed_radius_nearest_neighbor_jax(X, d_lim, h, cells):

  N = X.shape[0]               ## number of particles
  # n = jnp.ceil(d_lim/h).astype(int)             ## number of grid cells in each direction
  # n = int((d_lim / h) + ((d_lim % h) != 0))              ## number of grid cells in each direction
  # nb_cells = n*n*n

  nb_cells = cells.shape[0]


  points_to_cells = find_cell_vec(X, h, nb_cells**(1/3))

  cells_to_points = find_points_in_cell_vec(np.arange(nb_cells), points_to_cells)

  # cells_to_points_count = find_points_in_cell_count_vec(np.arange(nb_cells), points_to_cells)

  neighbor_points = []
  neighbor_dists = []

  for i in range(N):
    neighbor_cells = cells[points_to_cells[i]]

    neigh_ids = jnp.concatenate([cells_to_points[c] for c in neighbor_cells])
    neigh_ids = neigh_ids[(neigh_ids > -1) & (neigh_ids != i)]

    neigh_dists = distance_vec(X[i], X[neigh_ids], d_lim)

    neigh_ids = neigh_ids[neigh_dists <= h]
    neigh_dists = neigh_dists[neigh_dists <= h]

    # neigh_ids, neigh_dists = distances_from_point_i_to_cell_vec(i, neighbor_cells, X, cells_to_points, d_lim, h)

    # neighbor_points.append(np.asarray(neigh_ids.flatten()))
    # neighbor_dists.append(np.asarray(neigh_dists.flatten()))

    neighbor_points.append(np.asarray(neigh_ids))
    neighbor_dists.append(np.asarray(neigh_dists))

  return neighbor_points, neighbor_dists


