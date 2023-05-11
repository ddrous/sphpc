def euler_explicit_advection(positions, velocities, forces, dt):
    """ Euler time stepping scheme """
    velocities += dt * forces
    positions += dt * velocities
    return positions, velocities