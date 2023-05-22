import numpy as np





class SPHKernel():
    def __init__(self, smoothing_length, normalisation_constant):
        self.smoothing_length = smoothing_length
        self.normalisation_constant = normalisation_constant

    def apply(self, *args, **kwargs):
        raise NotImplementedError("Implement this method in a subclass")




class BSplineKernel(SPHKernel): #    see SPH Book page 314
    pass




class WendellKernel(SPHKernel):
    pass



# class CubicKernel(SPHKernel):
#     def __init__(self, smoothing_length, normalisation_constant):
#         super().__init__(smoothing_length, normalisation_constant)
    
#     def apply(self, distances, neighbor_ids):
#         """ Calculate the density of each particle """
#         nb_particles = distances.shape[0]
#         densities = np.zeros(nb_particles)

#         for i in range(nb_particles):
#             for j_in_list, j in enumerate(neighbor_ids[i]):
#                 q = distances[i][j_in_list] / self.smoothing_length
#                 if q <= 1:
#                     tmp = -(1 - q)

#         return densities


class Poly6Kernel(SPHKernel):
    def __init__(self, smoothing_length, normalisation_constant):
        super().__init__(smoothing_length, normalisation_constant)
    
    def apply(self, distances, neighbor_ids):
        """ Calculate the density of each particle """
        nb_particles = distances.shape[0]
        densities = np.zeros(nb_particles)

        for i in range(nb_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                densities[i] += self.normalisation_constant * (
                    self.smoothing_length**2
                    -
                    distances[i][j_in_list]**2
                )**3

        return densities





class SpikyKernel(SPHKernel):
    def __init__(self, smoothing_length, normalisation_constant):
        super().__init__(smoothing_length, normalisation_constant)

    def apply(self, positions, pressures, densities, distances, neighbor_ids):
        """ Calculate the pressure force of each particle """

        nb_particles = distances.shape[0]
        forces = np.zeros_like(positions)

        # Drop the element itself
        neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
        distances = [ np.delete(x, 0) for x in distances]

        for i in range(nb_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                # Pressure force
                forces[i] += self.normalisation_constant * (
                    -
                    (
                        positions[j]
                        -
                        positions[i]
                    ) / distances[i][j_in_list]
                    *
                    (
                        pressures[j]
                        +
                        pressures[i]
                    ) / (2 * densities[j])
                    *
                    (
                        self.smoothing_length
                        -
                        distances[i][j_in_list]
                    )**2
                )

        return forces
    


class ViscousKernel(SPHKernel):
    def __init__(self, smoothing_length, normalisation_constant):
        super().__init__(smoothing_length, normalisation_constant)

    def apply(self, velocities, densities, distances, neighbor_ids):
        """ Calculate the viscous force of each particle """
        nb_particles = distances.shape[0]
        forces = np.zeros_like(velocities)


        # Drop the element itself
        neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
        distances = [ np.delete(x, 0) for x in distances]

        for i in range(nb_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                # Viscous force
                forces[i] += self.normalisation_constant * (
                    (
                        velocities[j]
                        -
                        velocities[i]
                    ) / densities[j]
                    *
                    (
                        self.smoothing_length
                        -
                        distances[i][j_in_list]
                    )
                )

        return forces
    


class LennardJonesModel():
    def __init__(self, influence_range, reference_potential, p1, p2):
        self.influence_range = influence_range
        self.reference_potential = reference_potential
        self.p1 = p1
        self.p2 = p2

    def apply(self, positions, bd_positions, bd_distances, neighbor_ids):
        """
        Calculate the force applied by boundary points on each particle 
        See Voileu. 2012, equation (6.156)
        """
        nb_particles = positions.shape[0]
        forces = np.zeros_like(positions)

        for i in range(nb_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):

                forces[i] += self.reference_potential * (
                    (
                        self.influence_range / bd_distances[i][j_in_list]
                    ) ** self.p2
                    -
                    (
                        self.influence_range / bd_distances[i][j_in_list]
                    ) ** self.p1
                ) * ((bd_positions[j] - positions[i]) / bd_distances[i][j_in_list]**2)

        return forces



