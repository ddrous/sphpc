""" Steps for simulation

1. Import geometry from usda file
2. create arrays for position, velocity, and forces
3. Define all kernels
3. add particles from source
4. TIMESTEPPING LOOP find neighbourhood information
5. Calculate densities, pressures, 
6. Advect the particles and their velocities
7. check for collision and enforce boundary conditions
8. Export to OpenVDB file

"""

