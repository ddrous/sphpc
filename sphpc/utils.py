import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def random_name(length=5):
    "Make random names to identify runs"
    name = ""
    for _ in range(length):
        name += str(random.randint(0, 9))
    return name


def make_dir(path):
    "Make a directory if it doesn't exist"
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.mkdir(path)


plt.style.use('bmh')
sns.set(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
plt.style.use("dark_background")

## Wrapper function for matplotlib and seaborn
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, y_scale='linear', **kwargs):
    if ax==None: 
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_yscale(y_scale)
    ax.legend()
    plt.tight_layout()
    return ax


def muller_ns_eos(densities, base_density, isotropic_exponent):
    """ 
    Closure for NS (equation of state): Calculate the pressure of each particle 
    See Muller et al. 2003, equation 11
    """
    pressures = isotropic_exponent * (densities - base_density)

    return pressures

def lind_ns_eos(densities, base_density, isotropic_exponent):
    """ 
    Closure for NS (equation of state): Calculate the pressure of each particle 
    See Lind et al. 2020, equation 3.3
    """
    pressures = isotropic_exponent * (densities - base_density)

    return pressures


# def init_weakly_compressible_problem(geometry, nb_particles=1):
#     """ Initialize the position and velocity arrays for the weakly compressible problem """
#     positions = geometry.sources[0].points[:nb_particles]
#     velocities = np.zeros_like(positions)

#     return positions, velocities

def add_particles(geometry_source, velocities=None, nb_particles=1):
    """ Add particles to the simulation """
    if velocities is not None:
        nb_particles = velocities.shape[0]
    else:
        velocities = np.zeros((nb_particles, 3))

    positions = geometry_source.points[:nb_particles]

    return positions, velocities


def find_neighbours(positions, query_positions, smoothing_length):
    """ Find the neighbours of each particle """
    from sklearn import neighbors

    neighbor_ids, distances = neighbors.KDTree(
        positions,
    ).query_radius(
        query_positions,
        smoothing_length,
        return_distance=True,
        sort_results=True,
    )

    return neighbor_ids, distances


def visualize_flow(plt, positions, bd_positions, dot_size, fig_size):
    """ Visualize the flow in matplotlib """

    plt.scatter(
        positions[:, 0],
        positions[:, 2],
        s=dot_size,
        c=positions[:, 2],
        cmap="Wistia_r",
    )
    # plt.xlim(np.max(bd_positions[:,0]))
    # plt.ylim(np.max(bd_positions[:,2]))
    # plt.xticks([], [])
    # plt.yticks([], [])
    plt.tight_layout()
    plt.draw()
    plt.pause(1e-4)
    plt.clf()

    return plt