import jax.numpy as jnp
from jax import jit, value_and_grad, random, vmap, tree_map

import optax

import chex
import typing_extensions
from typing import NamedTuple

import warnings

from seql.agents.agent_utils import Memory
from seql.agents.base import Agent, LoglikelihoodFn, LogpriorFn

Params = chex.ArrayTree
Prior = chex.ArrayTree
Optimizer = NamedTuple


# https://github.com/deepmind/optax/blob/252d152660300fc7fe22d214c5adbe75ffab0c4a/optax/_src/transform.py#L35
class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""
    trace: chex.ArrayTree


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array):
        ...


class LossFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array,
                 model_fn: ModelFn) -> float:
        ...


class BeliefState(NamedTuple):
    params: Params
    prior: Prior
    opt_states: TraceState


class Info(NamedTuple):
    ...


def bootstrap_sampling(key, nsamples):
    def sample(key):
        return random.randint(key, (), 0, nsamples)

    keys = random.split(key, nsamples)
    return vmap(sample)(keys)


class EnsembleAgent(Agent):

    def __init__(self,
                 loglikelihood: LoglikelihoodFn,
                 model_fn: ModelFn,
                 nensembles: int,
                 logprior: LogpriorFn = lambda params: 0.,
                 nepochs: int = 20,
                 min_n_samples: int = 1,
                 buffer_size: int = jnp.inf,
                 obs_noise: float = 0.1,
                 beta: float = 0.1,
                 optimizer: Optimizer = optax.adam(1e-2),
                 is_classifier: bool = False):

        super(EnsembleAgent, self).__init__(is_classifier)

        assert min_n_samples <= buffer_size

        self.memory = Memory(buffer_size)
        self.model_fn = model_fn

        def loss_fn(params: Params,
                 x: chex.Array,
                 y: chex.Array):

            ll = loglikelihood(params,
                                x, y,
                                model_fn)
            lp = logprior(params)
            return -(ll + lp)

        self.loss_fn = loss_fn
        self.loglikelihood = loglikelihood
        self.logprior = logprior

        self.value_and_grad_fn = jit(value_and_grad(self.loss_fn))
        self.nensembles = nensembles
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.beta = beta
        self.nepochs = nepochs
        self.min_n_samples = min_n_samples
        self.obs_noise = obs_noise

    def init_state(self,
                   params: Params,
                   prior: Prior):
        opt_states = vmap(self.optimizer.init)(params)
        return BeliefState(params, prior, opt_states)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert self.buffer_size >= len(x)
        x_, y_ = self.memory.update(x, y)
        
        if len(x_) < self.min_n_samples:
            warnings.warn("There should be more data.", UserWarning)
            info = Info(False, -1, jnp.inf)
            return belief, info

        vbootstrap = vmap(bootstrap_sampling, in_axes=(0, None))

        keys = random.split(key, self.nensembles)
        indices = vbootstrap(keys, len(x_))

        x_ = vmap(lambda x, index: x[index], in_axes=(None, 0))(x_, indices)
        y_ = vmap(lambda x, index: x[index], in_axes=(None, 0))(y_, indices)
       
        def train(params, opt_state, x, y):
            for _ in range(self.nepochs):
                print(x.shape)
                loss, grads = self.value_and_grad_fn(params, x, y)
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            return params, opt_state

        vtrain = vmap(train, in_axes=(0, 0, 0, 0))
        params, opt_states = vtrain(belief.params, belief.opt_states, x_, y_)

        return BeliefState(params, belief.prior, opt_states), Info()

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        sample_key, key = random.split(key)
        index = random.randint(sample_key, (), 0, self.nensembles)
        params = tree_map(lambda prior, params: self.beta * prior[index] + params[index],
                          belief.prior, 
                          belief.params)
        return params