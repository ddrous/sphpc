
#%%


""" 
Algorithm to learn the signed distance based on a USDA mesh of the boundary
    1) for each normal and the corresponding point
    2) sample N points between the point and Lxpoint in two directions: the normal and the non-normal
    3) train the NN on the outside points with their loss (positive)
    4) do the same for the inside points with their own loss (negative)
    5) evaluate testerror on all boundary points
    6) Visualise the 0-levelset with Pyvista (results are not great)
"""



import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import os
import matplotlib.pyplot as plt
import pyvista as pv
import skimage

from sphpc import USDAGeom, make_dir, plot


#%%

EXPERIMENET_ID = "LearningSDF"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)
KEY = jax.random.PRNGKey(42)     ## Use same random points for all iterations

SURFACE_ID = 0   ## id of the boundary to learn
SAMPLING_FACTOR = 2

SAMPLING_DIST = 15
SAMPLING_SIZE = 100


PRINT_EVERY = 20

INIT_LR = 1e-3
MAX_EPOCHS = 1000
TOLERANCE = 1e-5

#%%

geometry = USDAGeom("meshes/glass.usda")
geometry.visualize(how="matplotlib");


#%%

ground_truth = geometry.boundaries[SURFACE_ID]
# ground_truth = geometry.sources[SURFACE_ID]

points = jnp.array([ground_truth.points[i] for i in ground_truth.faces])
normals = ground_truth.normals


## Data for testing
x_bd = ground_truth.points

## Data for training
def make_data(point, normal, sampling_dist, sampling_size, key):
    """ Make data for training in and opposite the direction of the normals """
    point = point[jnp.newaxis, :]
    normal = normal[jnp.newaxis, :]

    t = jax.random.uniform(key=key, shape=(sampling_size, 1), minval=0, maxval=sampling_dist)

    x_in = point - t*normal
    dist_in = -jnp.linalg.norm(x_in-point, axis=-1)[:, jnp.newaxis]

    x_out = point + t*normal
    dist_out = jnp.linalg.norm(x_out-point, axis=-1)[:, jnp.newaxis]

    return x_in, dist_in, x_out, dist_out

make_data_vec = jax.vmap(make_data, in_axes=(0, 0, None, None, 0), out_axes=(0))

res = make_data_vec(points, normals, SAMPLING_DIST, SAMPLING_SIZE, jax.random.split(KEY, num=points.shape[0]))

data_fn_vec = jax.vmap(make_data, in_axes=(0, 0, None, None, 0), out_axes=(0))

def make_data_vec(points, normals, sampling_dist, sampling_size, key):
    keys = jax.random.split(key, num=points.shape[0])

    data = data_fn_vec(points, normals, sampling_dist, sampling_size, keys)

    return [jnp.reshape(arr, (-1, arr.shape[-1])) for arr in data]


#%%

class MLP(nn.Module):
    input_size: int=3
    nb_layers:int=3
    nb_neurons_per_layer:int=20

    @nn.compact
    def __call__(self, x):                              ## TODO no normalisation
        y = nn.Dense(self.input_size)(x)
        y = nn.Dense(self.nb_neurons_per_layer)(y)
        for _ in range(0, self.nb_layers-1):
            y = nn.tanh(y)
            y = nn.Dense(self.nb_neurons_per_layer)(y)
        return nn.Dense(1)(y)

def init_flax_params(net:nn.Module):
    init_data = jnp.ones((10,3))
    params = net.init(KEY, init_data)
    print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

sdf = MLP()

print("Surrogate sdf archtecture: ")
params = init_flax_params(sdf)


#%%


total_steps = MAX_EPOCHS     ## number of optimisation steps

## Optimizer
scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.4):0.1, int(total_steps*0.8):0.1})

optimizer = optax.adam(learning_rate=scheduler)
# optimizer = optax.sgd(learning_rate=scheduler)

## Flax training state
state = train_state.TrainState.create(apply_fn=sdf.apply,
                                        params=params,
                                        tx=optimizer)



#%%


def loss_fn_train(params, x_in, dist_in, x_out, dist_out):

    ## Error in
    dist_in_pred = sdf.apply(params, x_in)
    agr_in = jnp.abs(dist_in_pred-dist_in)
    # pen_in = optax.l2_loss((dist_in_pred>0).astype(float))

    ## Error out
    dist_out_pred = sdf.apply(params, x_out)
    agr_out = jnp.abs(dist_out_pred-dist_out)
    # pen_out = optax.l2_loss((dist_out_pred<0).astype(float))

    return jnp.mean(agr_in) + jnp.mean(agr_out)

@jax.jit
def loss_fn_test(params, x_bd):
    dist_bd_pred = sdf.apply(params, x_bd)
    agr_bd = jnp.abs(dist_bd_pred)

    return jnp.mean(agr_bd)


#%%

@jax.jit
def train_step(state, x_in, dist_in, x_out, dist_out):
    loss, grads = jax.value_and_grad(loss_fn_train)(state.params, x_in, dist_in, x_out, dist_out)
    state = state.apply_gradients(grads=grads)

    return state, loss

# train_step_vec = jax.jit(jax.vmap(train_step, in_axes=(None, 0, 0, 0, 0), out_axes=(0)))

train_losses = []
test_losses = []


#%%

if len(os.listdir(DATAFOLDER)) > 2:
    print("Found existing network, loading & training")
    state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="sdf_checkpoint_", target=state)

else:
    print("Training from scratch")


epoch = 1
L = SAMPLING_DIST
subkey = KEY
test_losses.append(TOLERANCE+1)

while test_losses[-1] > TOLERANCE and epoch < MAX_EPOCHS+1:

    x_in, dist_in, x_out, dist_out = make_data_vec(points, normals, sampling_dist=L, sampling_size=SAMPLING_SIZE, key=KEY)
    state, loss = train_step(state, x_in, dist_in, x_out, dist_out)

    train_losses.append(loss)
    test_losses.append(loss_fn_test(state.params, x_bd))

    if epoch<=3 or epoch%PRINT_EVERY==0:
        print("Epoch: %-5d      TrainLoss: %.6f     TestLoss: %.6f" % (epoch, train_losses[-1], test_losses[-1]))

    L /= SAMPLING_FACTOR
    epoch += 1


checkpoints.save_checkpoint(DATAFOLDER, prefix="sdf_checkpoint_", target=state, step=state.step, overwrite=True)
print("Training done, saved network")


#%%

fig, ax = plt.subplots(1, 1, figsize=(6*1,4))

plot(train_losses[:], label='Train', x_label='epochs', y_scale="log", ax=ax)
plot(test_losses[1:], label='Test', x_label='epochs', y_scale="log", title="MAE", ax=ax);


# %%

## Marching cubes for visualisation

## Test function: a sphere with radius 1
def shpere(x):
    return -1. + np.linalg.norm(x, axis=-1)

# create a uniform grid to sample the function with
n = 100
x_min, y_min, z_min = -1.5, -1.5, -2
grid = pv.UniformGrid(
    dimensions=(n, n, n),
    spacing=(abs(x_min)*2 / n, abs(y_min)*2 / n, abs(z_min)*2 / n),
    origin=(x_min, y_min, z_min),
)

## Scalar values for the isosurface
values = np.asarray(sdf.apply(state.params, grid.points)).flatten()
# values = shpere(grid.points)
mesh = grid.contour(1, values, method='marching_cubes', rng=[-TOLERANCE, TOLERANCE])

## Plot the mesh with distance-based colouring 
dist = np.linalg.norm(mesh.points, axis=1)
mesh.plot(scalars=dist, smooth_shading=True, specular=1, cmap="plasma", show_scalar_bar=False)
