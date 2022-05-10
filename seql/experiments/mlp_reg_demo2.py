import chex
import jax.numpy as jnp
from jax import random, jit, vmap

import optax
import haiku as hk
from flax.core import frozen_dict
from matplotlib import pyplot as plt
from seql.agents.base import Agent

from seql.agents.bfgs_agent import BFGSAgent
from seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from seql.agents.ensemble_agent import EnsembleAgent
from seql.agents.sgd_agent import SGDAgent
from seql.agents.sgmcmc_sgld_agent import SGLDAgent
from seql.environments.base import make_mlp, make_regression_mlp_environment
from seql.environments.sequential_data_env import SequentialDataEnvironment
from seql.experiments.experiment_utils import run_experiment
from seql.utils import mean_squared_error
from seql.experiments.plotting import colors, plot_regression_posterior_predictive, sort_data

plt.style.use("seaborn-poster")

model_fn = None

def logprior_fn(params):

    return 0.

def loglikelihood_fn(params, x, y, model_fn):
    return -mean_squared_error(params, x, y, model_fn)

def initialize_params(agent_name, **kwargs):
    
    key = random.PRNGKey(233)
    def get_params(key):
        nfeatures = kwargs["nfeatures"]
        transformed = kwargs["transformed"]
        dummy_input = jnp.zeros([1, nfeatures])
        params = transformed.init(key, dummy_input)
        return params
    if agent_name == "ensemble":
        keys = random.split(key, 8)
        trainable = vmap(get_params)(keys)
        keys = random.split(random.PRNGKey(9), 8)
        baseline = trainable = vmap(get_params)(keys)
        params = frozen_dict.freeze(
            {"params": {"baseline": baseline,
                        "trainable": trainable
                        }
                })
    else: 

        params = get_params(key)
    return (params,)

def callback_fn(agent: Agent,
                env: SequentialDataEnvironment,
                agent_name: str,
                **kwargs):

    timesteps = kwargs["timesteps"]

    if "subplot_idx" not in kwargs and kwargs["t"] not in kwargs["timesteps"]:
        return

    nrows, ncols  = kwargs["nrows"], kwargs["ncols"]
    t = kwargs["t"]
    
    if "subplot_idx" not in kwargs:
        subplot_idx = timesteps.index(t) + kwargs["idx"] * ncols + 1
    else:
        subplot_idx = kwargs["subplot_idx"]

    fig = kwargs["fig"]

    '''if subplot_idx % ncols != 1:
        ax = fig.add_subplot(nrows,
                             ncols,
                             subplot_idx,
                             sharey=plt.gca(),
                             sharex=plt.gca())
    else:'''
    ax = fig.add_subplot(nrows,
                         ncols,
                         subplot_idx)
    belief = kwargs["belief"]

    X_line = jnp.linspace(-1.2, 1.2, 1000).reshape(-1, 1)

    outputs = env.true_model(X_line)

    outs = agent.posterior_predictive_mean_and_var(random.PRNGKey(0),
                                                   belief,
                                                   X_line,
                                                   200,
                                                   100)
    plot_regression_posterior_predictive(ax,
                                         env.X_test,
                                         env.y_test,
                                         jnp.squeeze(X_line),
                                         outs,
                                         outputs,
                                         agent_name,
                                         t=t)
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=32)
    else:
        ax.set_title("t={}".format(t), fontsize=32)

    plt.tight_layout()
    plt.savefig("jaks.png")
    plt.show()


def main():
    global model_fn
    key = random.PRNGKey(0)
    model_key, env_key, run_key = random.split(key, 3)
    ntrain = 20
    ntest = 20
    batch_size = 2
    obs_noise = 1.
    hidden_layer_sizes = [5, 5]
    nfeatures = 1
    ntargets = 1
    temperature = 1.

    net_fn = make_mlp(model_key,
                      nfeatures,
                      ntargets,
                      temperature,
                      hidden_layer_sizes)

    transformed = hk.without_apply_rng(hk.transform(net_fn))

    assert temperature > 0.0

    def forward(params: chex.Array, x: chex.Array):
        return transformed.apply(params, x) / temperature

    model_fn = jit(forward)

    env = lambda batch_size: make_regression_mlp_environment(env_key,
                                                             nfeatures,
                                                             ntargets,
                                                             ntrain,
                                                             ntest,
                                                             temperature=1.,
                                                             hidden_layer_sizes=hidden_layer_sizes,
                                                             train_batch_size=batch_size,
                                                             test_batch_size=batch_size,
                                                             )

    nsteps = 10

    buffer_size = ntrain

    optimizer = optax.adam(1e-2)

    nepochs = 20
    sgd = SGDAgent(loglikelihood_fn,
                   model_fn,
                   logprior_fn,
                   optimizer=optimizer,
                   obs_noise=obs_noise,
                   nepochs=nepochs,
                   buffer_size=buffer_size)

    optimizer = optax.adam(1e-2)
       
    batch_sgd = SGDAgent(loglikelihood_fn,
                        model_fn,
                        logprior_fn,
                         optimizer=optimizer,
                         obs_noise=obs_noise,
                         buffer_size=buffer_size,
                         nepochs=nepochs * nsteps)

    nsamples, nwarmup = 500, 200
    nuts = BlackJaxNutsAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        nsamples=nsamples,
        nwarmup=nwarmup,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    dt = 1e-5
    sgld = SGLDAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        dt=dt,
        batch_size=batch_size,
        nsamples=nsamples,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    # tau = 1.
    # strength = obs_noise / tau

    bfgs = BFGSAgent(loglikelihood_fn,
                     model_fn,
                     logprior_fn,
                     obs_noise=obs_noise,
                     buffer_size=buffer_size)

    def energy_fn(params, x, y):
        logprob = loglikelihood_fn(params, x, y, model_fn)
        logprob += logprior_fn(params)
        return -logprob/len(x)

    nensemble =  8

    ensemble = EnsembleAgent(loglikelihood_fn,
                             model_fn,
                             nensemble,
                             logprior_fn,
                             nepochs,
                             optimizer=optimizer)

    agents = {
        "nuts": nuts,
        "sgld": sgld,
        "sgd": sgd,
        "bfgs": bfgs,
        #"ensemble": ensemble,
    }

    batch_agents = agents.copy()
    batch_agents["sgd"] = batch_sgd
    
    timesteps = list(range(nsteps))

    nrows = len(agents)
    ncols = len(timesteps) + 1
    njoint = 10
    nsamples_input, nsamples_output = 1, 1

    run_experiment(run_key,
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
                   batch_agents=agents,
                   callback_fn=callback_fn,
                   obs_noise=obs_noise,
                   timesteps=list(range(nsteps)),
                   nfeatures=nfeatures,
                   transformed=transformed)


if __name__ == "__main__":
    main()
