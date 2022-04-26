import jax.numpy as jnp
from jax import random

import optax
from jaxopt import ScipyMinimize

from matplotlib import pyplot as plt

from seql.agents.bayesian_lin_reg_agent import BayesianReg
from seql.agents.bfgs_agent import BFGSAgent
from seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from seql.agents.kf_agent import KalmanFilterRegAgent
from seql.agents.laplace_agent import LaplaceAgent
from seql.agents.lbfgs_agent import LBFGSAgent
from seql.agents.sgd_agent import SGDAgent
from seql.agents.sgmcmc_sgld_agent import SGLDAgent
from seql.environments.base import make_evenly_spaced_x_sampler, \
    make_random_poly_regression_environment
from seql.experiments.experiment_utils import run_experiment
from seql.experiments.plotting import plot_regression_posterior_predictive
from seql.utils import mean_squared_error


plt.style.use("seaborn-poster")


def model_fn(w, x):
    return x @ w

def logprior_fn(params):
    strength = 0.1
    return strength * jnp.sum(params ** 2)

def loglikelihood_fn(params, x, y, model_fn):
    return -mean_squared_error(params, x, y, model_fn)


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

    outs = agent.posterior_predictive_mean_and_var(random.PRNGKey(0),
                                                   kwargs["belief"],
                                                   env.X_test[kwargs["t"]],
                                                   10,
                                                   5)
    plot_regression_posterior_predictive(ax,
                                         outs,
                                         env,
                                         agent_name,
                                         t=kwargs["t"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=32)
    else:
        ax.set_title("t={}".format(kwargs["t"]), fontsize=32)
    plt.tight_layout()
    plt.savefig("jaks.png")
    plt.show()


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["degree"] + 1
    mu0 = jnp.zeros((nfeatures, 1))
    if agent_name in ["exact bayes", "kf"]:
        mu0 = jnp.zeros((nfeatures, 1))
        Sigma0 = jnp.eye(nfeatures)
        initial_params = (mu0, Sigma0)
    else:
        initial_params = (mu0,)

    return initial_params


def main():
    key = random.PRNGKey(0)

    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    ntrain = 50
    ntest = 50
    batch_size = 5
    obs_noise = 1.

    env_key, run_key = random.split(key)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                                     degree,
                                                                     ntrain,
                                                                     ntest,
                                                                     obs_noise=obs_noise,
                                                                     train_batch_size=batch_size,
                                                                     test_batch_size=batch_size,
                                                                     x_test_generator=x_test_generator)

    nsteps = 10

    buffer_size = ntrain

    kf = KalmanFilterRegAgent(obs_noise=obs_noise)

    bayes = BayesianReg(buffer_size=buffer_size,
                        obs_noise=obs_noise)
                        
    batch_bayes = BayesianReg(buffer_size=ntrain,
                              obs_noise=obs_noise)

    optimizer = optax.adam(1e-1)

    nepochs = 4
    sgd = SGDAgent(loglikelihood_fn,
                   model_fn,
                   logprior_fn,
                   optimizer=optimizer,
                   obs_noise=obs_noise,
                   nepochs=nepochs,
                   buffer_size=buffer_size)

    batch_sgd = SGDAgent(loglikelihood_fn,
                        model_fn,
                        logprior_fn,
                         optimizer=optimizer,
                         obs_noise=obs_noise,
                         buffer_size=buffer_size,
                         nepochs=nepochs * nsteps)

    nsamples, nwarmup = 200, 100
    nuts = BlackJaxNutsAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        nsamples=nsamples,
        nwarmup=nwarmup,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    batch_nuts = BlackJaxNutsAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        nsamples=nsamples * nsteps,
        nwarmup=nwarmup,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    dt = 1e-4
    sgld = SGLDAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        dt=dt,
        batch_size=batch_size,
        nsamples=nsamples,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    dt = 1e-5
    batch_sgld = SGLDAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        dt=dt,
        batch_size=batch_size,
        nsamples=nsamples * nsteps,
        obs_noise=obs_noise,
        buffer_size=buffer_size)

    # tau = 1.
    # strength = obs_noise / tau

    bfgs = BFGSAgent(loglikelihood_fn,
                     model_fn,
                     logprior_fn,
                     obs_noise=obs_noise,
                     buffer_size=buffer_size)

    lbfgs = LBFGSAgent(loglikelihood_fn,
                        model_fn,
                        logprior_fn,
                       obs_noise=obs_noise,
                       history_size=buffer_size)

    def energy_fn(params, x, y):
        logprob = loglikelihood_fn(params, x, y, model_fn)
        logprob += logprior_fn(params)
        return -logprob

    solver = ScipyMinimize(fun=energy_fn, method="BFGS")
    laplace = LaplaceAgent(solver,
                        loglikelihood_fn,
                        model_fn,
                        logprior_fn,
                           obs_noise=obs_noise,
                           buffer_size=buffer_size)


    agents = {
        "kf": kf,
        "exact bayes": bayes,
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
        "sgld": sgld,
    }

    batch_agents = {
        "kf": kf,
        "exact bayes": batch_bayes,
        "sgd": batch_sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
    }

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
                   callback_fn=callback_fn,
                   degree=degree,
                   obs_noise=obs_noise,
                   timesteps=timesteps,
                   batch_agents=batch_agents
                   )

    env = lambda _: make_random_poly_regression_environment(env_key,
                                                            degree,
                                                            ntrain,
                                                            ntest,
                                                            obs_noise=obs_noise,
                                                            x_test_generator=x_test_generator)

    timesteps = list([1, 2, 5, 9, 19, 39])
    ncols = len(timesteps)
    run_experiment(run_key,
                   agents,
                   env,
                   initialize_params,
                   batch_size,
                   ntrain,
                   ntrain,
                   nsamples,
                   njoint,
                   nrows,
                   ncols,
                   callback_fn=callback_fn,
                   degree=degree,
                   obs_noise=obs_noise,
                   timesteps=timesteps)


if __name__ == "__main__":
    main()