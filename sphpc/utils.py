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


def visualise_sph_trajectory(trajs, vels, videoname, duration=10, vmin=None, vmax=None):

    import pyvista as pv
    # pv.start_xvfb()    ## To avoid seg fault in X-server
    pv.set_plot_theme("document")
    sargs = dict(height=0.25, width=0.035, vertical=True)
    pv.global_theme.font.size = 16
    pv.global_theme.font.label_size = 11


    speeds = np.linalg.norm(vels, axis=-1)

    mesh = pv.wrap(trajs[0])
    mesh.point_data["traj"] = trajs[0]
    mesh.point_data["speeds"] = speeds[0]

    plt = pv.Plotter()
    # Open a movie file
    nbframes = trajs.shape[0]
    plt.open_movie(videoname, framerate=nbframes/duration)

    # Add initial mesh
    if vmin==None:
        vmin = np.min(speeds)
    if vmax==None:
        vmax = np.max(speeds)


    plt.add_mesh(mesh, scalars="speeds", clim=[vmin, vmax], render_points_as_spheres=True, point_size=5, show_scalar_bar=True, scalar_bar_args=sargs, cmap="coolwarm")

    # plt.view_xy()

    plt.show_grid()
    plt.show_axes()

    x_max, y_max, z_max = np.max(trajs[:,:,0]), np.max(trajs[:,:,1]), np.max(trajs[:,:,2])

    plt.camera_position = [(2*x_max, -2.1*y_max, 2*z_max), (x_max/2., y_max/2., z_max/2.4), (0.0, 0.0, 0.1)]

    plt.show(auto_close=False)  # only necessary for an off-screen movie

    # Run through each frame
    plt.write_frame()  # write initial data


    # Update scalars on each frame
    for i in range(nbframes):
        ### Make sure field[i] is properly orderd first
        mesh.point_data["trajs"] = trajs[i]
        mesh.point_data["speeds"] = speeds[i]
        plt.add_text(f"Frame: {i+1} / {nbframes}", name='time-label', font_size=14, shadow=True, font='courier', position='upper_right')
        plt.write_frame()  # Write this frame

    # Be sure to close the plotter when finished
    plt.close()
