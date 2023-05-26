
from functools import lru_cache
import numpy as np
from collections import namedtuple


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

@lru_cache(maxsize=None)
def construct_cells(n):

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


"""
@Brief: This function computes the neighbors of each particle in a periodic domain
@Input: X: (N, 3) array of  particles positions at wich we query
@d_lim: period of the cubic domain
@h: query radius
"""
def periodic_fixed_radius_nearest_neighbor(X, d_lim, h):

  N = X.shape[0]               ## number of particles
  n = int(np.ceil(d_lim/h))             ## number of grid cells in each direction

  ## This can be easily vectorised
  def find_cell(x):
    return compute_id(int(x[0]/h), int(x[1]/h), int(x[2]/h), n)
  # find_cell_vec = np.vectorize(find_cell) ## TODO: vectorise this

  cells = construct_cells(n)

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

    nei_ids = []
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
