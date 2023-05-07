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
from sklearn import neighbors
from tqdm import tqdm


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
CONSTANT_FORCE = np.array([[0.0, -0.1]])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2_500
ADD_PARTICLES_EVERY = 50

FIGURE_SIZE = (4, 6)
PLOT_EVERY = 6
SCATTER_DOT_SIZE = 2_000

NORMALIZATION_DENSITY = ((315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9))

NORMALIZATION_PRESSURE_FORCE = -(45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)

NORMALIZATION_VISCOUS_FORCE = (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)











def main():
    """ Main function """

    # Create geometry from cube or usda file
    # geometry = USDAGeom("beer.usda")
    geometry = CubeGeom(DOMAIN_WIDTH=40, DOMAIN_HEIGHT=80)

    # Create arrays for position, velocity, and forces
    densities, positions, velocities = init_weakly_compressible_problem(geometry, nb_particles=1)

    # Define custom kernels
    density_kernel = Poly6Kernel(SMOOTHING_LENGTH, NORMALIZATION_DENSITY)

    pressure_kernel = SpikyKernel(SMOOTHING_LENGTH, NORMALIZATION_PRESSURE_FORCE)

    viscosity_kernel = ViscoKernel(SMOOTHING_LENGTH, NORMALIZATION_VISCOUS_FORCE)



    for iter in tqdm(range(N_TIME_STEPS)):

        ## Add particles from source (make sure particles go into the fluid domain)
        new_vels = np.array([[-3.0, -15.0], [-3.0, -15.0],[-3.0, -15.0],]) 
        positions, velocities = add_particles(geometry.source, vels=new_vels)

        ## Find neighbourhood information
        neighbours_ids, distances = find_neighbours(positions, SMOOTHING_LENGTH)


        ## Calculate densities and pressures
        densities = density_kernel.apply(distances, neighbours_ids)
        pressures = pressure_closure(densities, BASE_DENSITY, ISOTROPIC_EXPONENT)

        ## CaLculate pressure and viscous forces
        pressure_forces = pressure_kernel.apply(positions, pressures, densities, distances, neighbours_ids)

        viscous_forces = viscosity_kernel.apply(velocities, densities, distances, neighbours_ids)

        forces = CONSTANT_FORCE + (pressure_forces + viscous_forces) / densities


        ## Euler step
        positions, velocities = advect_fields(positions, velocities, forces, TIME_STEP_LENGTH)


        ## Enfore boundary conditions
        positions, velocities = enforce_boundary_conditions(geometry.boundary, DAMPING_COEFFICIENT)


        ## Plot or write to OpenVDB file
        if iter % PLOT_EVERY == 0:
            visualize_particles(plt, positions, SCATTER_DOT_SIZE, geometry.boundary.xmax, geometry.boundary.ymax)
