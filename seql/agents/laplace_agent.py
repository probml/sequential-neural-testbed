import jax.numpy as jnp
from jax import hessian, tree_map

import distrax

import chex
from typing import Any, NamedTuple, Optional
from functools import partial

import warnings

from seql.agents.agent_utils import Memory
from seql.agents.base import Agent, LoglikelihoodFn, LogpriorFn, ModelFn

JaxOptSolver = Any
Params = Any
Info = NamedTuple


class BeliefState(NamedTuple):
    mu: Params
    Sigma: Params = None


class Info(NamedTuple):
    ...


class LaplaceAgent(Agent):

    def __init__(self,
                 solver: JaxOptSolver,
                 loglikelihood: LoglikelihoodFn,
                 model_fn: ModelFn,
                 logprior: LogpriorFn = lambda params: 0.,
                 min_n_samples: int = 1,
                 buffer_size: int = 0,
                 obs_noise: float = 0.01,
                 is_classifier: bool = False):
        super(LaplaceAgent, self).__init__(is_classifier)

        self.memory = Memory(buffer_size)
        self.solver = solver
        self.model_fn = model_fn

        def loss_fn(params: Params,
                 x: chex.Array,
                 y: chex.Array):

            ll =  loglikelihood(params,
                                x, y,
                                self.model_fn)
            lp = logprior(params)
            return -(ll + lp)

        self.loss_fn = loss_fn
        self.obs_noise = obs_noise
        self.min_n_samples = min_n_samples
        self.buffer_size = buffer_size

    def init_state(self,
                   mu: chex.Array,
                   Sigma: Optional[chex.Array] = None):
        return BeliefState(mu, Sigma)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        x_, y_ = self.memory.update(x, y)

        if len(x_) < self.min_n_samples:
            warnings.warn("There should be more data.", UserWarning)
            return belief, None

        params, info = self.solver.run(belief.mu,
                                       x=x_,
                                       y=y_)
        partial_loss_fn = partial(self.loss_fn,
                                    x=x_,
                                    y=y_)

        Sigma = hessian(partial_loss_fn)(params)
        return BeliefState(params, tree_map(jnp.squeeze, Sigma)), info

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1),
                                                       Sigma)
        theta = mvn.sample(seed=key)
        theta = theta.reshape(mu.shape)
        return theta