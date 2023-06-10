import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp


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


def visualise_sph_trajectory(trajs, scals, videoname, duration=10, domain_lim=None, vmin=None, vmax=None):

    import pyvista as pv
    # pv.start_xvfb()    ## To avoid seg fault in X-server
    pv.set_plot_theme("document")

    scalars_name = scals[0]
    scalars = scals[1]

    # Min and max for scalar bar
    if vmin==None:
        vmin = np.min(scalars)
    if vmax==None:
        vmax = np.max(scalars)

    if domain_lim is None:
        x_max, y_max, z_max = np.max(trajs[:,:,0]), np.max(trajs[:,:,1]), np.max(trajs[:,:,2])
    else:
        x_max, y_max, z_max = domain_lim, domain_lim, domain_lim

    ## New camera position and focus on the center of the domain
    cam_pos = [(2*x_max, -2.1*y_max, 2*z_max), (x_max/2., y_max/2., z_max/2.4), (0.0, 0.0, 0.1)]


    ## Scalar bar arguments
    sbar_args = dict(height=0.25, width=0.035, 
                 vertical=True, 
                 position_x=0.9, position_y=0.4,
                 title=scalars_name,
                 title_font_size=22,
                 label_font_size=12,
                 shadow=True,
                 font_family="times")

    ## Mesh args
    mesh_args = dict(clim=[vmin, vmax],
                    scalars="speed", 
                    render_points_as_spheres=True,
                    point_size=5, 
                    show_scalar_bar=scalars_name!="", scalar_bar_args=sbar_args, 
                    cmap="coolwarm")

    ## Arguments for the grid wrapping the args
    grid_args = dict(bounds=[0., domain_lim, 0., domain_lim, 0., domain_lim],
                     grid="front",
                     xtitle="", 
                     ytitle="", 
                     ztitle="")

    ## Text arguments to be added to each frame
    text_args = dict(name='time-label', 
                    font_size=14, 
                    shadow=True, 
                    font='courier', 
                    position='lower_right')

    mesh = pv.wrap(np.copy(trajs[0]))
    mesh.points[...] = trajs[0]
    mesh.point_data["speed"] = scalars[0]


    pl = pv.Plotter()

    # Open a movie file
    nbframes = trajs.shape[0]
    pl.open_movie(videoname, framerate=(nbframes-1)/duration)

    pl.add_text(f"Frame: {1} / {nbframes}", **text_args)
    pl.add_mesh(mesh, **mesh_args)
    pl.show_grid(**grid_args)
    pl.show_axes()

    # pl.enable_eye_dome_lighting()
    # pl.show(auto_close=False)  # only necessary for an off-screen movie

    pl.show(auto_close=False, cpos=cam_pos)  # necessary for notebook inline plotting

    # Write the initial frame
    pl.write_frame()

    # Update coordinates and scalars on each frame
    for i in range(1, nbframes):

        mesh.points[...] = trajs[i]
        mesh.point_data["speed"] = scalars[i]
 
        # pl.add_text(f"Frame: {i+1} / {nbframes}", **text_args)
        pl.add_text("Frame: %3d / %3d"%(i+1, nbframes), **text_args)
        pl.show_grid(**grid_args)

        pl.write_frame()

    pl.close()



def flatten_params(params):
    tree_def = jax.tree_util.tree_structure(params)

    pytree = jax.tree_util.tree_leaves(params)
    flat_params = jnp.concatenate([x.flatten() for x in pytree])
    flat_shapes = [jnp.shape(x) for x in pytree]

    return flat_params, flat_shapes, tree_def


def unflatten_params(flat_params, flat_shapes, tree_def):
    pytree = []
    i = 0
    for shape in flat_shapes:
        size = np.prod(shape)
        pytree.append(jnp.reshape(flat_params[i:i+size], shape))
        i += size

    return jax.tree_util.tree_unflatten(tree_def, pytree)
