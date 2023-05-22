def euler_explicit_advection(positions, velocities, forces, dt):
    """ Euler time stepping scheme """
    velocities += dt * forces
    positions += dt * velocities
    return positions, velocities



def half_verlet_scheme(X, V, F, dt):
    """ Verlet time stepping scheme """
    V += 0.5 * dt * F
    X += dt * V
    return X, V
