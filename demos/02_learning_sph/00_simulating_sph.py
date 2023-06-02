# %%
import numpy as np
from tqdm import tqdm

from sphpc import *

# np.random.seed(42)
EXPERIMENET_ID = "3D"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
# make_dir(DATAFOLDER)

#params:

IC = "taylor-green"         ## Taylor-Green IC
# IC = "random"         ## TODO Change this and add proper Taylor-Green IC
# method = "AV_neg_rel" 

T = 3
T_SAVE = 0   #initial time for saving
PRINT_EVERY = 1

V_INIT = 1.0   #initial magnitude of Taylor-Green velocity
c = 0.9157061661168617
h = 0.2
α = 0.45216843078299573
β = 0.3346233846532608
γ = 1.0                     ## TODO Equal 7 in the paper (see fig 1)
θ = 0.00430899795067121
dt = 0.4 * h / c

pi = np.pi
D_LIM = 2.*pi    ## domain limit accors all axis

D = 3
HALF_RES = 5;  ## produces grid of 2^halfres x 2^halfres x 2^halfres number of particles

cube = CubeGeom(x_lim=D_LIM, y_lim=D_LIM, z_lim=D_LIM, halfres=HALF_RES)

N = cube.meshgrid.shape[0]
m = (D_LIM)**D / N       ## constant mass of each particle

print("Weakly Compressible SPH")
print(" -Number of particles: ", N)
print(" -Number of time steps: ", T)

## Not the same sigma as in the paper (see eq.14 Woodward 2023)
# sigma = (10. / (7. * pi * h * h)); #2D normalizing factor
sigma = 1 / (pi * (h**3))  #3D normalizing factor


# %%

def W(r, h):
  q = r / h   
  if (q > 2.):
    return 0
  if (q > 1.):
    return (sigma * (2. - q)**3 / 4.)
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)))

# H(r) = (d W / d r) / r
# Remember: dW(r)/dr = r \times H(r) is a vector
def H(r, h):
  q = r / h
  if (q > 2.):
    return 0
  if (q > 1.):
    return (-3. * sigma * (2. - q)**2 / (4. * h * r))
  return (sigma * (-3. + 9. * q / 4.) / (h * h))


## Equation of state
def P(rho):
  return (c**2 * (rho**γ - 1.) / γ)


## Speed of sound of particle rho
def compute_c(rho, c, γ):
  return c * rho**(0.5*(γ - 1))

## Artificial viscosity
def compute_Π(Xij, Vij, ρi, ρj, α, β, h, c):
  c_bar = (compute_c(ρi, c, γ) + compute_c(ρj, c, γ)) / 2.
  ρ_bar = (ρi + ρj) / 2.

  if (Xij.T@Vij < 0.):
    μ = h * (Xij.T@Vij) / (Xij.T@Xij + 0.01*(h**2))
    Π = (-α*c_bar*μ + β*(μ**2)) / ρ_bar
  else:
    Π = 0.

  return Π

## Density
def compute_densities(neighbor_ids, distances, h):
  N = len(neighbor_ids)
  ρ = np.zeros((N,))

  for i in range(N):
      ρ[i] += m * W(0., h)     ## i is its own neighbor
      for j_, j in enumerate(neighbor_ids[i]):
          ρ[i] += m * W(distances[i][j_], h)

  return ρ


## Acceleration force
def compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c):

  N = len(neighbor_ids)

  # Drop the element itself (if using KDTree)
  # neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
  # distances = [ np.delete(x, 0) for x in distances]

  F = np.zeros_like(X)

  for i in range(N):
    vali = P(ρ[i]) / (ρ[i]**2)

    for j_, j in enumerate(neighbor_ids[i]):

      ## Compute artificial viscosity
      Xij = X[i] - X[j]
      Vij = V[i] - V[j]
      for d in range(D):    ## TODO Optimize this
        while (Xij[d] > D_LIM/2.): Xij[d] -= D_LIM
        while (Xij[d] < -D_LIM/2.): Xij[d] += D_LIM
      Πij = compute_Π(Xij, Vij, ρ[i], ρ[j], α, β, h, c)

      valj = P(ρ[j]) / (ρ[j]**2)

      ## Add up all forces
      F[i] += -m*(vali + valj + Πij) * H(distances[i][j_], h) * Xij ## TODO External forcing?

  ke = 0.5 * np.mean(ρ * np.sum(V**2, axis=-1))
  # F += θ * V / ke
  F += θ * (V - np.mean(V, axis=0)) / (2*ke)    ## TODO Change to the above as in the paper

  return F

def check_nans(X, V, ρ):
  print(f"Min Max of X: {np.min(X):.2} {np.max(X):.2}", f"\t\tMin Max of V: {np.min(V):.2} {np.max(V):.2}")

  if (np.any(np.isnan(X)) or np.any(np.isnan(V)) or np.any(np.isnan(ρ))):
    print("NaNs found in X, V, or ρ")
    exit(1)


# %%

## Algorithm begins here

print("**************** Simulating the particle flow ***************")

X = cube.meshgrid
V = V_INIT * np.ones_like(X)
# V[N//2:, 1] *= -1   ## Default IC: Blue particles to right, red to left

if IC == "random":
  X += 0.0005 * (np.random.uniform(0., 1., size=cube.meshgrid.shape) - 0.5)
  V = V_INIT * np.random.normal(0., 1., size=X.shape)

X = np.mod(X, D_LIM)

if IC == "taylor-green":
  V[:, 0] = V_INIT * np.sin(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2])
  V[:, 1] = -V_INIT * np.cos(X[:, 0]) * np.sin(X[:, 1]) * np.cos(X[:, 2])
  V[:, 2] = 0.

trajs, vels = np.zeros((T+1, N, D)), np.zeros((T+1, N, D))

trajs[0, :, :] = X
vels[0, :, :] = V

rhos = np.zeros((T+1, N))

# neighbor_ids, distances = find_neighbours(X, X, h)


cells = construct_cells_for_nn_search(D_LIM, h)
neighbor_ids, distances = periodic_fixed_radius_nearest_neighbor(X, D_LIM, h, cells)

# cells = construct_cells_for_nn_search_jax(D_LIM, h)
# neighbor_ids, distances = periodic_fixed_radius_nearest_neighbor_jax(X, D_LIM, h, cells)

ρ = compute_densities(neighbor_ids, distances, h)
rhos[0, :] = ρ


for t in tqdm(range(1, T+1)):

  # neighbor_ids, distances = find_neighbours(X, X, h)
  neighbor_ids, distances = periodic_fixed_radius_nearest_neighbor(X, D_LIM, h, cells)
  # neighbor_ids, distances = periodic_fixed_radius_nearest_neighbor_jax(X, D_LIM, h, cells)

  ρ = compute_densities(neighbor_ids, distances, h)
  F = compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c)

  ## Verlet advection scheme part 1
  X, V = half_verlet_scheme(X, V, F, dt)
  X = np.mod(X, D_LIM)

  ## Check nans before continuing
  # check_nans(X, V, ρ)

  neighbor_ids, distances = periodic_fixed_radius_nearest_neighbor(X, D_LIM, h, cells)

  # ρ = compute_densities(neighbor_ids, distances, h)
  F = compute_acc_forces(X, V, ρ, neighbor_ids, distances, α, β, h, c)

  ## Verlet advection scheme part 2
  _, V = half_verlet_scheme(X, V, F, dt)

  rhos[t, :] = ρ
  trajs[t, :, :] = X
  vels[t, :, :] = V

  # if (t % PRINT_EVERY == 1):
    # print(f"Time step: ", t)
  # print(neighbor_ids[:100])
  print("\n")
  print([(np.sum(dist <= 0.), dist.shape[0]) for dist in distances])
  print("Maximum of forces: ", np.max(F))
  print("Minimum of forces: ", np.min(F))

print("****************  Simulation COMPLETE  *************")


# %%

print("**************** Visualising the particle flow ***************")

speeds = np.linalg.norm(vels, axis=-1)

red_blue = np.zeros_like(speeds)
red_blue[:, N//2:] = 1

visualise_sph_trajectory(trajs, 
                        # ("Speeds", speeds),
                        ("", red_blue),
                        DATAFOLDER+"trajectory.mp4", 
                        duration=15, 
                        domain_lim=D_LIM, 
                        vmin=None, vmax=None)

print("****************  Visualisation COMPLETE  *************")

# %%

## Save files

print(" ****************** Saving data files ***********************")

pos_path = DATAFOLDER+f"trajs_N{N}_T{T}_dt{dt}_ts{T_SAVE}_h{h}_{IC}_θ{θ}.npy"
vel_path = DATAFOLDER+f"vels_N{N}_T{T}_dt{dt}_ts{T_SAVE}_h{h}_{IC}_θ{θ}.npy"
rho_path = DATAFOLDER+f"rhos_N{N}_T{T}_dt{dt}_ts{T_SAVE}_h{h}_{IC}_θ{θ}.npy"

np.save(pos_path, trajs[T_SAVE:,:,:])
np.save(vel_path, vels[T_SAVE:,:,:])
np.save(rho_path, rhos[T_SAVE:,:])

print("****************  Saving COMPLETE  *************")




# %%

