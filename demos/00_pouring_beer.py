# %%

""" 

SPH Simulation of Beer Pouring: "Particle-Based Fluid Simulation for Interactive Applications", Matthias Müller, David Charypar, and Markus Gross, SCA 2003


Steps for simulation

1. Import geometry from usda file
2. Create arrays for position, velocity, and forces
3. Define all kernels
3. Add particles from source
4. In the timesteppiong loop, find neighbourhood information
5. Calculate densities, pressures, 
6. Advect the particles and their velocities
7. check for collision and enforce boundary conditions
8. Export to OpenVDB file

"""


## Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sphpc import *


## Constants

MAX_PARTICLES = 125
DOMAIN_WIDTH = 40
DOMAIN_HEIGHT = 80

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([[0.0, 0.0, -0.1]])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2_5
ADD_PARTICLES_EVERY = 50

FIGURE_SIZE = (4, 6)
PLOT_EVERY = 6
SCATTER_DOT_SIZE = 2_000

NORMALIZATION_DENSITY = (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9)

NORMALIZATION_PRESSURE_FORCE = -(45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)

NORMALIZATION_VISCOUS_FORCE = (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)


INFLUENCE_RANGE = 0.5
REFERENCE_POTENTIAL = 5 * 1 * 9.81 * 1e-10  ## See Voileu. 2012, equation (6.156)
P1 = 4
P2 = 2








# Create geometry from cube or usda file
geometry = USDAGeom("meshes/glass.usda")
geometry.visualize()

# Create arrays for position, velocity, and forces
positions, velocities = add_particles(geometry.boundaries[0], nb_particles=0)

# Define custom kernels
density_kernel = Poly6Kernel(SMOOTHING_LENGTH, NORMALIZATION_DENSITY)

pressure_kernel = SpikyKernel(SMOOTHING_LENGTH, NORMALIZATION_PRESSURE_FORCE)

viscosity_kernel = ViscousKernel(SMOOTHING_LENGTH, NORMALIZATION_VISCOUS_FORCE)


bd_positions = geometry.boundaries[0].points
boundary_model = LennardJonesModel(INFLUENCE_RANGE, REFERENCE_POTENTIAL, P1, P2)


plt.style.use("dark_background")
plt.figure(figsize=FIGURE_SIZE, dpi=160)

for iter in tqdm(range(N_TIME_STEPS)):

    ## Add particles from source (make sure particles go into the fluid domain)
    new_vels = np.array([[-3.0, 0.0, -15.0], [-3.0, 0.0, -15.0],[-3.0, 0.0, -15.0],]) 
    new_positions, new_velocities = add_particles(geometry.sources[0], new_vels)

    # nb_particles += new_velocities.shape[0]
    positions = np.concatenate((positions, new_positions), axis=0)
    velocities = np.concatenate((velocities, new_velocities), axis=0)


    ## Find neighbourhood information
    neighbor_ids, distances = find_neighbours(positions, positions, SMOOTHING_LENGTH)


    ## Calculate densities and pressures
    densities = density_kernel.apply(distances, neighbor_ids)
    pressures = muller_ns_eos(densities, BASE_DENSITY, ISOTROPIC_EXPONENT)

    ## CaLculate pressure and viscous forces
    pressure_forces = pressure_kernel.apply(positions, pressures, densities, distances, neighbor_ids)

    viscous_forces = viscosity_kernel.apply(velocities, densities, distances, neighbor_ids)


    ## Enfore boundary conditions
    bd_neighbors_ids, bd_distances = find_neighbours(bd_positions, positions, SMOOTHING_LENGTH)

    bd_forces = boundary_model.apply(positions, bd_positions, bd_distances, bd_neighbors_ids)


    forces = CONSTANT_FORCE + bd_forces + (pressure_forces + viscous_forces) / densities[:, np.newaxis]


    ## Euler step
    positions, velocities = euler_explicit_advection(positions, velocities, forces, TIME_STEP_LENGTH)

    # positions, velocities = enforce_boundary_conditions(geometry.boundary, DAMPING_COEFFICIENT)


    ## Plot or write to OpenVDB file
    if iter % PLOT_EVERY == 0:
        plt = visualize_flow(plt, positions, bd_positions, SCATTER_DOT_SIZE, FIGURE_SIZE)
