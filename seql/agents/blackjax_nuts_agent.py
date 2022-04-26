import jax.numpy as jnp
from jax import jit, random, lax, tree_map

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import chex
import warnings
from typing import Any, NamedTuple

from seql.agents.agent_utils import Memory
from seql.agents.base import Agent, LoglikelihoodFn, LogpriorFn, ModelFn

Params = Any
Samples = Any
State = NamedTuple


class BeliefState(NamedTuple):
    state: State = None
    step_size: float = 0.
    inverse_mass_matrix: chex.Array = None
    samples: Samples = None


class Info(NamedTuple):
    ...


class NutsState(NamedTuple):
    # https://github.com/blackjax-devs/blackjax/blob/fd83abf6ce16f2c420c76772ff2623a7ee6b1fe5/blackjax/mcmc/integrators.py#L12
    position: chex.ArrayTree
    momentum: chex.ArrayTree = None
    potential_energy: float = None
    potential_energy_grad: chex.ArrayTree = None


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = random.split(rng_key, num_samples)
    final, states = lax.scan(one_step, initial_state, keys)

    return final, states


class BlackJaxNutsAgent(Agent):

    def __init__(self,
                 loglikelihood: LoglikelihoodFn,
                 model_fn: ModelFn,
                 nsamples: int,
                 nwarmup: int,
                 logprior: LogpriorFn = lambda params: 0.,
                 nlast: int = 10,
                 buffer_size: int = 0,
                 min_n_samples: int = 1,
                 obs_noise: float = 0.1,
                 is_classifier: bool = False):

        super(BlackJaxNutsAgent, self).__init__(is_classifier)

        if buffer_size == jnp.inf:
            buffer_size = 0

        assert min_n_samples <= buffer_size or buffer_size == 0
        self.memory = Memory(buffer_size)


        self.model_fn = model_fn

        def logprob_fn(params: Params,
                 x: chex.Array,
                 y: chex.Array):

            ll =  loglikelihood(params,
                                x, y,
                                self.model_fn)
            lp = logprior(params)
            return ll + lp

        self.logprob = logprob_fn
        self.loglikelihood = loglikelihood
        self.logprior = logprior

        self.nwarmup = nwarmup
        self.nlast = nlast
        self.nsamples = nsamples
        self.obs_noise = obs_noise
        self.buffer_size = buffer_size
        self.threshold = min_n_samples

    def init_state(self,
                   initial_position: Params):
        nuts_state = NutsState(initial_position)
        return BeliefState(nuts_state)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert self.buffer_size >= len(x)
        x_, y_ = self.memory.update(x, y)

        if len(x_) < self.threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, Info()

        @jit
        def partial_logprob(params):
            return self.logprob(params, x_, y_)

        warmup_key, sample_key = random.split(key)

        state = nuts.new_state(belief.state.position,
                               partial_logprob)

        kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(partial_logprob,
                                                                              step_size,
                                                                              inverse_mass_matrix)
        final, (step_size, inverse_mass_matrix), info = stan_warmup.run(warmup_key,
                                                                              kernel_generator,
                                                                              state,
                                                                              self.nwarmup)

        # Inference
        nuts_kernel = jit(nuts.kernel(partial_logprob,
                                      step_size,
                                      inverse_mass_matrix))

        final, states = inference_loop(sample_key,
                                       nuts_kernel,
                                       state,
                                       self.nsamples)

        belief_state = BeliefState(final,
                                   step_size,
                                   inverse_mass_matrix,
                                   tree_map(lambda x: x[-self.nlast:], states))
        return belief_state, Info()

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        return tree_map(lambda x: x.mean(axis=0), belief.samples.position)