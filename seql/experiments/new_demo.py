import jax.numpy as jnp

import jax

import flax.linen as nn
from plotnine import *
import optax
from sklearn.preprocessing import PolynomialFeatures
from seql.agents.base import Agent

from seql.agents.eekf_agent import EEKFAgent
from seql.agents.bfgs_agent import BFGSAgent
from seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from seql.agents.lbfgs_agent import LBFGSAgent
from seql.agents.sgd_agent import SGDAgent
from seql.agents.sgmcmc_sgld_agent import SGLDAgent
from seql.environments.base import make_classification_mlp_environment, make_random_poly_classification_environment
from seql.environments.sequential_classification_env import SequentialClassificationEnvironment
from seql.experiments.experiment_utils import run_experiment
from seql.experiments.plotting import sanity_plots
from seql.utils import cross_entropy_loss, train
from jsl.nlds.base import NLDS


class MLP(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x
                 ):
        x = nn.relu(nn.Dense(50)(x))
        x = nn.relu(nn.Dense(50)(x))
        x = nn.Dense(self.nclasses)(x)
        return jax.nn.softmax(x, axis=-1)


model = MLP(2)
model_fn = model.apply


def logprior_fn(params, strength=0.2):
    return 0.
    leaves = jax.tree_leaves(params)
    return -sum(jax.tree_map(lambda x: jnp.sum(x ** 2), leaves)) * strength


def loglikelihood_fn(params, x, y, model_fn):
    logprobs = model_fn(params, x)
    return -cross_entropy_loss(y, logprobs)


def callback_fn(agent: Agent,
                env: SequentialClassificationEnvironment,
                **kwargs):
    plots = sanity_plots(prior,
                 env,
                 agent,
                 kwargs["belief"])
    for x, y in plots.items():
        file_name = f"{x}{kwargs['t']}.png"
        ggsave(y, file_name)

prior = None

def main():
    global prior
    key = jax.random.PRNGKey(0)

    ntrain, ntest = 20, 20
    batch_size = 20
    nfeatures, nclasses = 2, 2

    env_key, experiment_key = jax.random.split(key, 2)
    obs_noise = 0.

    prior, env = make_classification_mlp_environment(env_key,
                                                     nfeatures,
                                                     nclasses,
                                                     ntrain,
                                                     ntest,
                                                     temperature=1.,
                                                     hidden_layer_sizes=[50, 50],
                                                     train_batch_size=batch_size,
                                                     test_batch_size=batch_size)

    is_classifier = True

    # initial random guess
    buffer_size = ntrain
    optimizer = optax.adam(1e-2)
    nepochs = 30

    sgd = SGDAgent(loglikelihood_fn,
                    model_fn,
                    logprior_fn,
                    nepochs=nepochs,
                    buffer_size=buffer_size,
                    obs_noise=obs_noise,
                    optimizer=optimizer,
                    is_classifier=is_classifier)
    
    batch = jnp.ones((1, nfeatures))
    variables = model.init(jax.random.PRNGKey(0), batch)
    belief = sgd.init_state(variables)

    nsteps = 1
    train(experiment_key,
          belief,
          sgd,
          env,
          nsteps,
          1,
          1,
          1,
          callback_fn)
    




if __name__ == "__main__":
    main()
