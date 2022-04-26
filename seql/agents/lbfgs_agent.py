import warnings
import jax.numpy as jnp

from jaxopt import LBFGS

import chex
from typing import Any, Callable, NamedTuple, Optional, Union
from seql.agents.agent_utils import Memory

from seql.agents.base import Agent, LoglikelihoodFn, LogpriorFn, ModelFn

Params = Any
AutoOrBoolean = Union[str, bool]


class BeliefState(NamedTuple):
    params: Params


class LBFGSAgent(Agent):

    def __init__(self,
                 loglikelihood: LoglikelihoodFn,
                 model_fn: ModelFn,
                 logprior: LogpriorFn = lambda params: 0.,
                 maxiter: int = 500,
                 maxls: int = 15,
                 history_size: int = 10,
                 buffer_size: int = jnp.inf,
                 min_n_samples: int = 1,
                 obs_noise: float = 0.1,
                 decrease_factor: float = 0.8,
                 increase_factor: float = 1.5,
                 tol: float = 1e-3,
                 implicit_diff_solve: Optional[Callable] = None,
                 condition: str = "strong-wolfe",
                 jit: AutoOrBoolean = "auto",
                 unroll: AutoOrBoolean = "auto",
                 has_aux: bool = False,
                 use_gamma: bool = True,
                 implicit_diff: bool = True,
                 verbose: bool = False,
                 is_classifier: bool = False):
        '''
        https://github.com/google/jaxopt/blob/53b539e6c5cee4c52262ce17d4601839422ffe87/jaxopt/_src/lbfgs.py#L148
        Attributes:
            fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
            has_aux: whether function fun outputs one (False) or more values (True).
            When True it will be assumed by default that fun(...)[0] is the objective.
            maxiter: maximum number of proximal gradient descent iterations.
            tol: tolerance of the stopping criterion.
            maxls: maximum number of iterations to use in the line search.
            decrease_factor: factor by which to decrease the stepsize during line search
            (default: 0.8).
            increase_factor: factor by which to increase the stepsize during line search
            (default: 1.5).
            history_size: size of the memory to use.
            use_gamma: whether to initialize the inverse Hessian approximation with
            gamma * I, see 'Numerical Optimization', equation (7.20).
            implicit_diff: whether to enable implicit diff or autodiff of unrolled
            iterations.
            implicit_diff_solve: the linear system solver to use.
            jit: whether to JIT-compile the optimization loop (default: "auto").
            unroll: whether to unroll the optimization loop (default: "auto").
            verbose: whether to print error on every iteration or not.
            Warning: verbose=True will automatically disable jit.
        Reference:
            Jorge Nocedal and Stephen Wright.
            Numerical Optimization, second edition.
            Algorithm 7.5 (page 179).
        '''
        super(LBFGSAgent, self).__init__(is_classifier)

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

        assert min_n_samples <= buffer_size

        self.memory = Memory(buffer_size)
        self.lbfgs = LBFGS(self.loss_fn,
                           has_aux,
                           maxiter,
                           tol,
                           condition,
                           maxls,
                           decrease_factor,
                           increase_factor,
                           history_size,
                           use_gamma,
                           implicit_diff,
                           implicit_diff_solve,
                           jit,
                           unroll,
                           verbose)
        self.buffer_size = buffer_size
        self.min_n_samples = min_n_samples
        self.obs_noise = obs_noise

    def init_state(self,
                   params: Params):
        return BeliefState(params)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert self.buffer_size >= len(x)
        x_, y_ = self.memory.update(x, y)

        if len(x_) < self.min_n_samples:
            warnings.warn("There should be more data.", UserWarning)
            return belief, None

        params, info = self.lbfgs.run(belief.params,
                                      x=x_,
                                      y=y_)
        return BeliefState(params), info

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        n = len(x)
        posterior_cov = self.obs_noise * jnp.eye(n)
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        return belief.params
