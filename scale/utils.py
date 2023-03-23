import jax
import jax.numpy as jnp
from jax.tree_util import Partial


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


# plt.style.use('bmh')
# sns.set(context='notebook', style='ticks',
#         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

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

