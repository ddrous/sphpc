import jax

@jax.jit
def euler_explicit_advection(positions, velocities, forces, dt):
    """ Euler time stepping scheme """
    velocities += dt * forces
    positions += dt * velocities
    return positions, velocities


@jax.jit
def half_verlet_scheme(X, V, F, dt):
    """ Verlet time stepping scheme """
    Vnew = V + 0.5 * dt * F
    return X + dt*Vnew, Vnew
