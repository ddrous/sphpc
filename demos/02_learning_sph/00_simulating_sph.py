# %%
import numpy as np
from sphpc import *

np.random.seed(42)
EXPERIMENET_ID = "3D"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

#params:

T = 400
t_save = 1   #initial time for saving
vmag = 1.0   #initial magnitude of TG velocity

c = 0.9157061661168617
h = 0.2
α = 0.45216843078299573
β = 0.3346233846532608
γ = 1.0
θ = 0.00430899795067121
cdt = 0.4
dt = cdt * h / c
p_gt = [c, 0.0]

pi = np.pi
DOMAIN_LIM = 2*pi

D = 3
halfres = 2; #produces grid of 2^halfres x 2^halfres x 2^halfres number of particles

cube = CubeGeom(x_lim=DOMAIN_LIM, y_lim=DOMAIN_LIM, z_lim=DOMAIN_LIM, halfres=halfres)

N = cube.meshgrid.shape[0]
m = (DOMAIN_LIM)**D / N       ## constant mass of each particle

## Not the same sigma as in the paper (see eq.14 Woodward 2023)
# sigma = (10. / (7. * pi * h * h)); #2D normalizing factor
sigma = 1/(pi*h**3)  #3D normalizing factor


# %%

def W(r, h):
  q = r / h   
  if (q > 2.):
    return 0
  if (q > 1.):
    return (sigma * (2. - q)**3 / 4.)
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)))   ## ERROR in

# H(r) = (d W / d r) / r
# Remember: dW(r)/dr = r \times H(r) is a vector
def H(r, h):
  q = r / h
  if (q > 2.):
    return 0
  if (q > 1.):
    return (-3. * sigma * (2. - q)**2 / (4. * h * r))
  return (sigma * (-3. + 9. * q / 4.) / (h * r))


## Equation of state
def P(rho):
  return (c**2 * (rho**γ - 1.) / γ)


def compute_ca(rho, c, γ):
  return c * rho**(0.5*(γ - 1))

def compute_Π(XX, VV, ρi, ρj, α, β, h, c):
  cj = compute_ca(ρj, c, γ)
  ci = compute_ca(ρi, c, γ)
  c_bar = (ci + cj)/2
  if (XX.T@VV < 0):
    μ = h*(XX.T@VV)/(np.sum(XX**2, axis=-1) + 0.01*(h**2))
    Π = (-α*c_bar*μ + β * (μ**2))/((ρi + ρj)/2)
  else:
    Π = 0.0
  return Π

def compute_densities(neighbor_ids, distances, h):
  ρ = np.zeros((len(neighbor_ids), ))
  for i in range(N):
      for j_in_list, j in enumerate(neighbor_ids[i]):
          ρ[i] += m * W(distances[i][j_in_list], h)

  return ρ


def compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c):

  # Drop the element itself
  neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
  distances = [ np.delete(x, 0) for x in distances]

  F = np.zeros_like(X)

  for i in range(N):
    val1 = P(ρ[i]) / (ρ[i]**2)
    for j_in_list, j in enumerate(neighbor_ids[i]):

      ## Compute artificial viscosity
      r_ij = X[i] - X[i]
      v_ij = V[i] - V[j]
      for d in range(D):
        while (r_ij[d] > DOMAIN_LIM/2.): r_ij[d] -= DOMAIN_LIM
        while (r_ij[d] < -DOMAIN_LIM/2.): r_ij[d] += DOMAIN_LIM
      Π = compute_Π(r_ij, v_ij, rhos[i], rhos[j], α, β, h, c)

      ## Add up all forces
      val2 = P(ρ[j]) / (ρ[j]**2)

      F[i] = -m*(val1 + val2 + Π) * H(distances[i][j_in_list], h) * r_ij ## No external forces for now TODO Add later

  return F


# %%

## Algorithm begins here
X = cube.meshgrid + 0.0005 * (np.random.uniform(0., 1., size=cube.meshgrid.shape) - 0.5)

X = np.mod(X, DOMAIN_LIM)
V = vmag * np.random.uniform(0., 1., size=X.shape)

traj, vels = np.zeros((T+1, N, D)), np.zeros((T+1, N, D))
traj[0:, :, :] = X
vels[0:, :, :] = V

rhos = np.zeros((T+1, N))
neighbor_ids, distances = find_neighbours(X, X, 16*h)
ρ = compute_densities(neighbor_ids, distances, h)
rhos[0, :] = ρ


for t in range(1, T+1):

  ## TODO Find neighbours in a periodic way: hash table
  neighbor_ids, distances = find_neighbours(X, X, h)

  ρ = compute_densities(neighbor_ids, distances, h)
  F = compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c)

  ## Verlet advection scheme part 1
  X, V = half_verlet_scheme(X, V, F, dt)
  X = np.mod(X, DOMAIN_LIM)


  ρ = compute_densities(neighbor_ids, distances, h)
  F = compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c)

  ## Verlet advection scheme part 2
  _, V = half_verlet_scheme(X, V, F, dt)

  rhos[t, :] = ρ
  traj[t, :, :] = X
  vels[t, :, :] = V

  print("Time step: ", t)


# %%


visualise_sph_trajectory(traj, vels, DATAFOLDER+"trajectory.mp4", duration=10, vmin=None, vmax=None)


# %%
