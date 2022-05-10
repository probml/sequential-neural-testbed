import jax.numpy as jnp
from jax import random, nn

import optax
from jaxopt import ScipyMinimize

from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from seql.agents.base import Agent

from seql.agents.bayesian_lin_reg_agent import BayesianReg
from seql.agents.bfgs_agent import BFGSAgent
from seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from seql.agents.ensemble_agent import EnsembleAgent
from seql.agents.kf_agent import KalmanFilterRegAgent
from seql.agents.laplace_agent import LaplaceAgent
from seql.agents.sgd_agent import SGDAgent
from seql.agents.sgmcmc_sgld_agent import SGLDAgent
from seql.environments.base import make_evenly_spaced_x_sampler, make_gaussian_sampler, \
    make_random_poly_regression_environment
from seql.environments.sequential_data_env import SequentialDataEnvironment
from seql.experiments.experiment_utils import run_experiment
from seql.experiments.plotting import plot_regression_posterior_predictive, sort_data
from seql.utils import mean_squared_error


plt.style.use("seaborn-poster")


def model_fn(w, x):
    print(w.shape, x.shape)
    return x @ w

def logprior_fn(params):
    strength = 0.
    return strength * jnp.sum(params ** 2)

def loglikelihood_fn(params, x, y, model_fn):
    return -mean_squared_error(params, x, y, model_fn)


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

    X_line = jnp.linspace(-1.2, 1.2, 1000)
    poly = PolynomialFeatures(3)
    phi = jnp.array(poly.fit_transform(X_line.reshape((-1, 1))), dtype=jnp.float32)


    #phi = env.X_test[:t+1]
    X_test, _ = sort_data(phi, phi)
    outputs = env.true_model(X_test)

    outs = agent.posterior_predictive_mean_and_var(random.PRNGKey(0),
                                                   belief,
                                                   X_test,
                                                   200,
                                                   100)
    plot_regression_posterior_predictive(ax,
                                         env.X_test,
                                         env.y_test,
                                         X_test[:, 1],
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


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["degree"] + 1
    mu0 = jnp.zeros((nfeatures, 1))
    if agent_name in ["exact bayes", "kf"]:
        mu0 = jnp.zeros((nfeatures, 1))
        Sigma0 = jnp.eye(nfeatures)
        initial_params = (mu0, Sigma0)
    elif agent_name =="ensemble":
        initializer = nn.initializers.glorot_normal()
        mu0 = initializer(random.PRNGKey(0), (8,  nfeatures, 1))
        prior = initializer(random.PRNGKey(2), (8,  nfeatures, 1))
        initial_params = (mu0, prior)
    else:
        initial_params = (mu0,)

    return initial_params


def main():
    key = random.PRNGKey(0)

    min_val, max_val = -0.6, 0.6
    scale = 1.

    x_train_generator = make_gaussian_sampler(0, scale)

    min_val, max_val = -1, 1
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    ntrain = 12
    ntest = 12
    batch_size = 3
    obs_noise = 1.

    env_key, run_key = random.split(key)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                                     degree,
                                                                     ntrain,
                                                                     ntest,
                                                                     obs_noise=obs_noise,
                                                                     train_batch_size=batch_size,
                                                                     test_batch_size=batch_size,
                                                                     x_train_generator=x_train_generator,
                                                                     x_test_generator=x_test_generator,
                                                                     shuffle=True)

    nsteps = 4

    buffer_size = 5

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
                         buffer_size=ntrain,
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

    batch_nuts = BlackJaxNutsAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        nsamples=nsamples * nsteps,
        nwarmup=nwarmup,
        obs_noise=obs_noise,
        buffer_size=ntrain)

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

    dt = 1e-5
    batch_sgld = SGLDAgent(
        loglikelihood_fn,
        model_fn,
        logprior=logprior_fn,
        dt=dt,
        batch_size=ntrain,
        nsamples=nsamples * nsteps,
        obs_noise=obs_noise,
        buffer_size=ntrain)

    # tau = 1.
    # strength = obs_noise / tau

    bfgs = BFGSAgent(loglikelihood_fn,
                     model_fn,
                     logprior_fn,
                     obs_noise=obs_noise,
                     buffer_size=buffer_size)
    
    batch_bfgs = BFGSAgent(loglikelihood_fn,
                    model_fn,
                    logprior_fn,
                    obs_noise=obs_noise,
                    buffer_size=ntrain)


    def energy_fn(params, x, y):
        logprob = loglikelihood_fn(params, x, y, model_fn)
        logprob += logprior_fn(params)
        return -logprob/len(x)

    solver = ScipyMinimize(fun=energy_fn, method="BFGS")
    laplace = LaplaceAgent(solver,
                        loglikelihood_fn,
                        model_fn,
                        logprior_fn,
                        obs_noise=obs_noise,
                        buffer_size=buffer_size)

    batch_laplace = LaplaceAgent(solver,
                    loglikelihood_fn,
                    model_fn,
                    logprior_fn,
                    obs_noise=obs_noise,
                    buffer_size=ntrain)
                    
    nensemble =  8

    ensemble = EnsembleAgent(loglikelihood_fn,
                             model_fn,
                             nensemble,
                             logprior_fn,
                             nepochs)

    agents = {
        
        "ensemble": ensemble,
        "laplace": laplace,
        "nuts": nuts,
        "sgld": sgld,
        "kf": kf,
        "exact bayes": bayes,
        "sgd": sgd,
        "bfgs": bfgs,
    }

    batch_agents = {
        "ensemble": ensemble,
        "kf": kf,
        "exact bayes": batch_bayes,
        "sgd": batch_sgd,
        "laplace": batch_laplace,
        "bfgs": batch_bfgs,
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


if __name__ == "__main__":
    main()