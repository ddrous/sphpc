def Euler(positions, velocities, forces, dt):
    """ Euler time stepping scheme """
    velocities += forces * dt
    positions += velocities * dt
    return positions, velocities