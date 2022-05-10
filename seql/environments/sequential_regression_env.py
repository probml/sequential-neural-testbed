import jax.numpy as jnp
from jax import random, jit, vmap

import chex
from typing import Tuple, Optional
from seql.agents.base import Agent

from seql.environments.sequential_data_env import SequentialDataEnvironment


Data = Tuple[chex.Array, chex.Array]

def gaussian_log_likelihood(err: chex.Array,
                            cov: chex.Array) -> float:
  """Calculates the Gaussian log likelihood of a multivariate normal."""
  first_term = len(err) * jnp.log(2 * jnp.pi)
  _, second_term = jnp.linalg.slogdet(cov)
  third_term = jnp.einsum('ai,ab,bi->i', err, jnp.linalg.pinv(cov), err)
  return -0.5 * (first_term + second_term + third_term)

def _kl_gaussian(mean_1: float,
                 std_1: float,
                 mean_2: float, std_2: float) -> float:
  """Computes the KL(P_1 || P_2) for P_1,P_2 univariate Gaussian."""
  log_term = jnp.log(std_2 / std_1)
  frac_term = (std_1 ** 2 + (mean_1 - mean_2) ** 2) / (2 * std_2 ** 2)
  return log_term + frac_term - 0.5

class SequentialRegressionEnvironment(SequentialDataEnvironment):
    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 test_mean : chex.Array,
                 test_cov: chex.Array,
                 true_model: chex.Array,
                 y_function: chex.Array,
                 train_batch_size: int,
                 obs_noise: float,
                 tau: int = 1,
                 key: Optional[chex.PRNGKey] = None):

        super().__init__(X_train,
                         y_train,
                         X_test,
                         true_model,
                         train_batch_size,
                         tau,
                         key)

        self.obs_noise = obs_noise
        self._num_train, self._input_dim = X_train.shape
        num_test_x_cache = len(X_test)

        self._tau = tau
        self._num_test_x_cache = num_test_x_cache
        self._y_train_function = y_function[:self._num_train]
        self._y_test__function = y_function[self._num_train:]
        self._test_mean = test_mean
        self._test_cov = test_cov
        self._noise_std = jnp.sqrt(obs_noise)

    @property
    def test_mean(self) -> chex.Array:
        return self._test_mean


    @property
    def test_cov(self) -> chex.Array:
        return self._test_cov



    def test_data(self, key: chex.PRNGKey) -> Tuple[Data, float]:
        """Generates test data and evaluates log likelihood under posterior.
        The test data that is output will be of length tau examples.
        We wanted to "pass" tau here but ran into jit issues.
        Args:
        key: Random number generator key.
        Returns:
        Tuple of data (with tau examples) and log-likelihood under posterior.
        """

        def sample_test_data(key: chex.PRNGKey) -> Tuple[Data, float]:
            x_key, fn_key, y_key =random.split(key, 3)

            # Sample tau x's from the testing cache for evaluation
            test_x_indices = random.randint(
                x_key, [self._tau], 0, self._num_test_x_cache)
            x_test = self.X_test[test_x_indices]
            chex.assert_shape(x_test, [self._tau, self._input_dim])

            # Sample the true function from the posterior mean
            nngp_mean = self._test_mean[test_x_indices, 0]
            chex.assert_shape(nngp_mean, [self._tau])
            nngp_cov = self._test_cov[jnp.ix_(test_x_indices, test_x_indices)]
            chex.assert_shape(nngp_cov, [self._tau, self._tau])
            sampled_fn = random.multivariate_normal(fn_key, nngp_mean, nngp_cov)
            y_noise = random.normal(y_key, [self._tau, 1]) * self._noise_std
            y_test = sampled_fn[:, None] + y_noise
            chex.assert_shape(y_test, [self._tau, 1])

            # Compute the log likelihood (under both posterior and noise)
            err = y_test - nngp_mean[:, None]
            chex.assert_shape(err, [self._tau, 1])
            cov = nngp_cov + self._noise_std ** 2 * jnp.eye(self._tau)
            chex.assert_shape(cov, [self._tau, self._tau])
            log_likelihood = gaussian_log_likelihood(err, cov)
            return (x_test, y_test), log_likelihood

        return jit(sample_test_data)(key)


    def evaluate_quality(self, key, agent: Agent, belief, num_enn_samples) -> float:
        """Computes KL estimate on mean functions for tau=1 only."""
        # Extract useful quantities from the gp sampler.
        x_test = self.X_test
        num_test = x_test.shape[0]
        posterior_mean = self.test_mean[:, 0]
        posterior_std = jnp.sqrt(jnp.diag(self.test_cov))
        posterior_std += 1e-6 # std_ridge

        # Compute the mean and std of ENN posterior

        enn_mean, enn_std = agent.posterior_predictive_mean_and_var(
                                    key,
                                    belief,
                                    x_test,
                                    num_enn_samples,
                                    1
                                    )
        # Compute the KL divergence between this and reference posterior
        batched_kl = jit(vmap(_kl_gaussian))
        kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean[:, 0], enn_std[:, 0])
        chex.assert_shape(kl_estimates, [num_test])
        kl_estimate = jnp.mean(kl_estimates)
        return kl_estimate