import jax.numpy as jnp
from jax import random

import optax
from matplotlib import pyplot as plt


from jsl.experimental.seql.agents.sgd_agent import SGDAgent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, \
    make_random_poly_regression_environment
from jsl.experimental.seql.utils import mean_squared_error, train


plt.style.use("seaborn-poster")


def model_fn(w, x):
    return x @ w

def logprior_fn(params):
    strength = 0.1
    return strength * jnp.sum(params ** 2)

def loglikelihood_fn(params, x, y, model_fn):
    return -mean_squared_error(params, x, y, model_fn)

kls = []

def callback_fn(**kwargs):
    global kls
    if kwargs["t"] == 0:
        kls.append([])
    kls[-1].append(kwargs["kl"])

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
    global kls
    key = random.PRNGKey(0)

    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    obs_noise = 1.

    taus = list(range(1, 11))
    colors = {1:"tab:blue", 10:"tab:red"}
    ntrain = 100
    ntest = 1000
    nsamples_input, nsamples_output = 1, 1
    test_batch_size = 1

    for rng_key, njoint in zip(random.split(key, len(taus)),
                                  taus):
        env_key, train_key = random.split(rng_key)
        nsteps = min(10, ntrain)
        train_batch_size = ntrain // nsteps
        env = make_random_poly_regression_environment(env_key,
                                                        degree,
                                                                        ntrain,
                                                                        ntest,
                                                                        obs_noise=obs_noise,
                                                                        train_batch_size=train_batch_size,
                                                                        test_batch_size=test_batch_size,
                                                                        x_test_generator=x_test_generator)

        
        buffer_size = ntrain
        optimizer = optax.adam(1e-3)
        nepochs = 4

        sgd = SGDAgent(loglikelihood_fn,
                    model_fn,
                    logprior_fn,
                    optimizer=optimizer,
                    obs_noise=obs_noise,
                    nepochs=nepochs,
                    buffer_size=buffer_size)
        params = jnp.zeros((degree + 1, 1))
        belief = sgd.init_state(params)
        train(train_key,
                belief,
                sgd,
                env,
                nsteps,
                nsamples_input,
                nsamples_output,
                njoint,
                callback_fn)

    y = [kl[-1] for kl in kls]        
    plt.plot(taus, y, '-o', c=colors[njoint])
    plt.xticks(taus)
    plt.xlabel("Order of Joint Predictive Distribution")
    plt.ylabel("Average KL Estimate")
    plt.savefig("njoint2.png")
        

if __name__ == "__main__":
    main()