# Kalman filter agent
import jax.numpy as jnp

import distrax

import chex
from typing import NamedTuple

from seql.agents.base import Agent
from jsl.lds.kalman_filter import LDS, kalman_filter


class BeliefState(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array


class Info(NamedTuple):
    mu_hist: chex.Array = None
    Sigma_hist: chex.Array = None


class KalmanFilterRegAgent(Agent):

    def __init__(self,
                 obs_noise: float = 1.,
                 return_history: bool = False,
                 is_classifier: bool = False):
        assert is_classifier == False
        super(KalmanFilterRegAgent, self).__init__(is_classifier)

        self.obs_noise = obs_noise
        self.return_history = return_history
        self.model_fn = lambda params, x: x @ params

    def init_state(self,
                   mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        *_, input_dim = x.shape
        *_, output_dim = y.shape

        A = jnp.eye(input_dim)
        Q = 0
        C = lambda t: x[t][None, ...]
        R = self.obs_noise * jnp.ones((1, 1))
        

        lds = LDS(A, C, Q, R,
                 jnp.squeeze(belief.mu),
                 belief.Sigma)

        mu, Sigma, _, _ = kalman_filter(lds,
                                        y,
                                        return_history=self.return_history)
        if self.return_history:
            history = (mu, Sigma)
            mu, Sigma = mu[-1], Sigma[-1]
            mu = mu.reshape((-1, output_dim))
            return BeliefState(mu, Sigma), Info(*history)
        
        return BeliefState(mu.reshape((-1, output_dim)), Sigma), Info()

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        n = len(x)
        posterior_cov = x @ belief.Sigma @ x.T + self.obs_noise * jnp.eye(n)
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu),
                                                       Sigma)
        theta = mvn.sample(seed=key).reshape(mu.shape)
        return theta
