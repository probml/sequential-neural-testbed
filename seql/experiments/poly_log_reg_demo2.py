import jax.numpy as jnp
from jax import random, tree_leaves, tree_map
from jax import nn

from matplotlib import pyplot as plt
import optax
from sklearn.preprocessing import PolynomialFeatures

from seql.agents.eekf_agent import EEKFAgent
from seql.agents.bfgs_agent import BFGSAgent
from seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from seql.agents.lbfgs_agent import LBFGSAgent
from seql.agents.sgd_agent import SGDAgent
from seql.agents.sgmcmc_sgld_agent import SGLDAgent
from seql.environments.base import make_random_poly_classification_environment
from seql.experiments.experiment_utils import run_experiment
from seql.experiments.plotting import sort_data
from seql.utils import cross_entropy_loss
from jsl.nlds.base import NLDS


def fz(x):
    return x


def model_fn(w, x):
    return nn.softmax(x @ w, axis=-1)


def logprior_fn(params, strength=0.):
    leaves = tree_leaves(params)
    return -sum(tree_map(lambda x: jnp.sum(x ** 2), leaves)) * strength


def loglikelihood_fn(params, x, y, model_fn):
    logprobs = model_fn(params, x)
    return -cross_entropy_loss(y, logprobs)


def print_accuracy(logprobs, ytest):
    ytest_ = jnp.squeeze(ytest)
    predictions = jnp.where(logprobs > jnp.log(0.5), 1, 0)
    print("Accuracy: ", jnp.mean(jnp.argmax(predictions, axis=-1) == ytest_))

def get_grid(min_x, max_x, min_y, max_y):
    # define the x and y scale
    x1grid = jnp.arange(min_x, max_x, 0.1)
    x2grid = jnp.arange(min_y, max_y, 0.1)

    # create all of the lines and rows of the grid
    xx, yy = jnp.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = jnp.hstack((r1, r2))
    return xx, yy, grid

def callback_fn(agent, env, agent_name, **kwargs):
    if "subplot_idx" not in kwargs and kwargs["t"] not in kwargs["timesteps"]:
        return
    elif "subplot_idx" not in kwargs:
        subplot_idx = kwargs["timesteps"].index(kwargs["t"]) + kwargs["idx"] * kwargs["ncols"] + 1
    else:
        subplot_idx = kwargs["subplot_idx"]

    ax = kwargs["fig"].add_subplot(kwargs["nrows"],
                                   kwargs["ncols"],
                                   subplot_idx)

    belief = kwargs["belief"]

    
    min_x, max_x = -3, 3
    min_y, max_y = -3, 3
    xx, yy, grid = get_grid(min_x, max_x, min_y, max_y)

    poly = PolynomialFeatures(kwargs["degree"])

    
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=32)
    else:
        ax.set_title("t={}".format(kwargs["t"]), fontsize=32)

    t = kwargs["t"]

    x, y = sort_data(env.X_test[:t + 1], env.y_test[:t + 1])
    colors = ["#86bbd8", "#E39191"]
    for cls, color in enumerate(colors):
        indices = jnp.argwhere(y == cls)
        # Plot training data
        ax.scatter(x[indices, 1],
                   x[indices, 2],
                   s=240.,
                   c=color,
                   edgecolor="black")
    n = xx.shape[0]
    
    x = poly.fit_transform(grid)
    
    for i in range(10):
        w = agent.sample_params(random.PRNGKey(i*42), belief)        
        pred = model_fn(w.reshape((10,2)), x)
        ax.contour(xx, yy, pred[:, 0].reshape((n, n)), jnp.array([0.5]))
    

    plt.tight_layout()
    plt.savefig("jakjs.png")


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["nfeatures"]
    nclasses = kwargs["nclasses"]

    mu0 = random.normal(random.PRNGKey(0), (nfeatures, nclasses))

    if agent_name == "eekf":
        mu0 = jnp.ravel(mu0)
        Sigma0 = jnp.eye(nfeatures * nclasses)
        initial_params = (mu0, Sigma0)
    else:
            
        initial_params = (mu0,)
    
    return initial_params


def main():
    key = random.PRNGKey(0)

    degree = 3
    ntrain, ntest = 12, 12
    batch_size = 3
    nsteps = 4
    nfeatures, nclasses = 2, 2

    env_key, experiment_key = random.split(key, 2)
    obs_noise = 0.
    env = lambda batch_size: make_random_poly_classification_environment(env_key,
                                                                         degree,
                                                                         ntrain,
                                                                         ntest,
                                                                         nfeatures=nfeatures,
                                                                         nclasses=nclasses,
                                                                         obs_noise=obs_noise,
                                                                         train_batch_size=batch_size,
                                                                         test_batch_size=batch_size,
                                                                         shuffle=False)

    buffer_size = ntrain

    input_dim = 10
    nparams = input_dim * nclasses
    Q = jnp.eye(nparams) * 1e-4
    R = jnp.eye(nclasses) * obs_noise

    is_classifier = True

    def eekf_model_fn(params, x):
        return model_fn(params.reshape((input_dim, nclasses)), x)

    # initial random guess
    nlds = NLDS(fz, eekf_model_fn, Q, R)
    eekf = EEKFAgent(nlds,
                     eekf_model_fn,
                     obs_noise,
                     is_classifier=is_classifier)

    
    #tau = 1.
    #strength = obs_noise / tau

    nepochs = 20
    optimizer = optax.adam(1e-2)

    sgd = SGDAgent(loglikelihood_fn,
    model_fn,
    logprior_fn,
    nepochs=nepochs,
    buffer_size=buffer_size,
    obs_noise=obs_noise,
    optimizer=optimizer,
    is_classifier=is_classifier)

    batch_sgd = SGDAgent(loglikelihood_fn,
    model_fn,
    logprior_fn,
    nepochs=nepochs * nsteps,
    buffer_size=buffer_size,
    obs_noise=obs_noise,
    optimizer=optimizer,
    is_classifier=is_classifier)



    nsamples, nwarmup = 500, 300
    nlast = 100
    nuts = BlackJaxNutsAgent(loglikelihood_fn,
    model_fn,
    nsamples,
    nwarmup,
    logprior_fn,
    obs_noise=obs_noise,
    buffer_size=buffer_size,
    is_classifier=is_classifier,
    nlast=nlast)

    batch_nuts = BlackJaxNutsAgent(loglikelihood_fn,
    model_fn,
    nsamples * nsteps,
    nwarmup,
    logprior_fn,
    obs_noise=obs_noise,
    buffer_size=buffer_size,
    is_classifier=is_classifier,
    nlast=nlast)


    dt = 1e-5

    sgld = SGLDAgent(loglikelihood_fn,
                     model_fn,
                     dt,
                     batch_size,
                     nsamples,
                     logprior_fn,
                     buffer_size=buffer_size,
                     obs_noise=obs_noise,
                     is_classifier=is_classifier)
    
    bfgs = BFGSAgent(loglikelihood_fn,
                     model_fn,
                     logprior_fn,
                     buffer_size=buffer_size,
                     obs_noise=obs_noise)

    lbfgs = LBFGSAgent(loglikelihood_fn,
    model_fn,
    logprior_fn,
    buffer_size=buffer_size,
    obs_noise=obs_noise)


    agents = {
        "eekf": eekf,
        "sgld": sgld,
        "sgd": sgd,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts":nuts,
    }

    batch_agents = {
        "eekf": eekf,
        "sgd": batch_sgd,
        "nuts": batch_nuts,
        "sgld": sgld,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
    }

    timesteps = list(range(nsteps))
    nrows = len(agents)
    ncols = len(timesteps)
    njoint = 10
    nsamples_input, nsamples_output = 1, 1
    run_experiment(experiment_key,
                   agents,
                   env,
                   initialize_params,
                   batch_size,
                   ntrain,
                   nsteps,
                   nsamples_input,
                   nsamples_output,
                   njoint,
                   nrows,
                   ncols,
                   callback_fn=callback_fn,
                   nfeatures=input_dim,
                   obs_noise=obs_noise,
                   batch_agents=batch_agents,
                   timesteps=timesteps,
                   degree=degree,
                   nclasses=nclasses)


if __name__ == "__main__":
    main()
