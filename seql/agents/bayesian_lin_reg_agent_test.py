"""Tests for jsl.sent.agents.bayesian_lin_reg_agent"""
import jax.numpy as jnp
from jax import random, vmap

import chex

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from seql.agents.bayesian_lin_reg_agent import BayesianReg
from seql.agents.kf_agent import KalmanFilterRegAgent
from seql.environments.base import make_evenly_spaced_x_sampler, make_random_poly_regression_environment
from seql.environments.sequential_regression_env import SequentialRegressionEnvironment
from seql.utils import train



kf_mean, kf_cov = None, None


def callback_fn(**kwargs):
    global kf_mean, kf_cov

    mu_hist = kwargs["info"].mu_hist
    Sigma_hist = kwargs["info"].Sigma_hist

    if kf_mean is not None:
        kf_mean = jnp.vstack([kf_mean, mu_hist])
        kf_cov = jnp.vstack([kf_cov, Sigma_hist])
    else:
        kf_mean = mu_hist
        kf_cov = Sigma_hist


bayes_mean, bayes_cov = None, None


def bayes_callback_fn(**kwargs):
    global bayes_mean, bayes_cov

    mu = kwargs["belief"].mu[None, ...]
    Sigma = kwargs["belief"].Sigma[None, ...]

    if bayes_mean is not None:
        bayes_mean = jnp.vstack([bayes_mean, mu])
        bayes_cov = jnp.vstack([bayes_cov, Sigma])
    else:
        bayes_mean = mu
        bayes_cov = Sigma


class BayesLinRegTest(parameterized.TestCase):

    @parameterized.parameters(itertools.product((4,), (1, 0, 5), (0.1,)))
    def test_init_state(self,
                        input_dim: int,
                        buffer_size: int,
                        obs_noise: float):
        output_dim = 1
        agent = BayesianReg(buffer_size, obs_noise)
        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim)
        belief = agent.init_state(mu, Sigma)

        chex.assert_shape(belief.mu, mu.shape)
        chex.assert_shape(belief.Sigma, Sigma.shape)

        assert agent.obs_noise == obs_noise
        assert agent.buffer_size == buffer_size

    @parameterized.parameters(itertools.product((0,),
                                                (10,),
                                                (2,),
                                                (10,),
                                                (0.1,)))
    def test_update(self,
                    seed: int,
                    ntrain: int,
                    input_dim: int,
                    buffer_size: int,
                    obs_noise: float):
        output_dim = 1

        agent = BayesianReg(buffer_size, obs_noise)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim)
        initial_belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, initial_belief, x, y)

        chex.assert_shape(belief.mu, (input_dim, output_dim))
        chex.assert_shape(belief.Sigma, (input_dim, input_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (2,),
                                                (10,),
                                                (0.1,)))
    def test_sample_params(self,
                           seed: int,
                           input_dim: int,
                           buffer_size: int,
                           obs_noise: float):
        output_dim = 1

        agent = BayesianReg(buffer_size, obs_noise)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        theta = agent.sample_params(key, belief)

        chex.assert_shape(theta, (input_dim, output_dim))


    @parameterized.parameters(itertools.product((0,),
                                                (10,),
                                                (2,),
                                                (10,),
                                                (5,),
                                                (10,),
                                                (0.1,)))
    def test_posterior_predictive_sample(self,
                                         seed: int,
                                         ntrain: int,
                                         input_dim: int,
                                         nsamples_params: int,
                                         nsamples_output: int,
                                         buffer_size: int,
                                         obs_noise: float,
                                         ):
        output_dim = 1

        agent = BayesianReg(buffer_size, obs_noise)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, ppd_key = random.split(key)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        samples = agent.posterior_predictive_sample(key, belief, x, nsamples_params, nsamples_output)
        chex.assert_shape(samples, (nsamples_params, ntrain, nsamples_output, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (5,),
                                                (2,),
                                                (10,),
                                                (10,),
                                                (0.1,)))
    def test_logprob_given_belief(self,
                                  seed: int,
                                  ntrain: int,
                                  input_dim: int,
                                  nsamples_params: int,
                                  buffer_size: int,
                                  obs_noise: float,
                                  ):
        output_dim = 1

        agent = BayesianReg(buffer_size, obs_noise)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        samples = agent.logprob_given_belief(logprob_key, belief, x, y, nsamples_params)
        chex.assert_shape(samples, (ntrain, output_dim))
        assert jnp.any(jnp.isinf(samples)) == False
        assert jnp.any(jnp.isnan(samples)) == False


    @parameterized.parameters(itertools.product((0,),
                                                (3, 5),
                                                (50,),
                                                (50,),
                                                (0.1,),
                                                (5,),
                                                (5,),
                                                (-10,),
                                                (10,)))
    def test_kf_vs_bayes(self,
                        seed,
                        degree,
                        ntrain,
                        ntest,
                        obs_noise,
                        train_batch_size,
                        test_batch_size,
                        min_val,
                        max_val):
        global kf_mean, kf_cov
        global bayes_mean, bayes_cov

        kf_mean, kf_cov = None, None
        bayes_mean, bayes_cov = None, None

        x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

        key = random.PRNGKey(seed)
        env = make_random_poly_regression_environment(key,
                                                      degree,
                                                      ntrain,
                                                      ntest,
                                                      obs_noise=obs_noise,
                                                      train_batch_size=train_batch_size,
                                                      test_batch_size=test_batch_size,
                                                      x_test_generator=x_test_generator)

        *_, input_dim = env.X_train.shape

        nsteps = 1
        mu0 = jnp.zeros(input_dim)
        Sigma0 = jnp.eye(input_dim) * 10.

        agent = KalmanFilterRegAgent(obs_noise,
                                  return_history=True)
        belief = agent.init_state(mu0, Sigma0)
        nsamples, njoint = 1, 1
        key = random.PRNGKey(seed)
        kf_belief, _ = train(key,
                             belief,
                             agent,
                             env,
                             nsteps=nsteps,
                             nsamples_output=nsamples,
                             nsamples_input=nsamples,
                             njoint=njoint,
                             callback=callback_fn)

        buffer_size = 1
        agent = BayesianReg(buffer_size, obs_noise)
        belief = agent.init_state(mu0.reshape((-1, 1)), Sigma0)
        key = random.PRNGKey(seed)

        bayes_belief, _ = train(key,
                             belief,
                             agent,
                             env,
                             nsteps=nsteps,
                             nsamples_output=nsamples,
                             nsamples_input=nsamples,
                             njoint=njoint,
                             callback=bayes_callback_fn)

        assert jnp.allclose(jnp.squeeze(kf_belief.mu),
                            jnp.squeeze(bayes_belief.mu),
                            atol=1e-2)

        assert jnp.allclose(kf_belief.Sigma,
                            bayes_belief.Sigma)

        '''# Posterior predictive distribution check
        v_ppd = vmap(self._posterior_preditive_distribution,
                     in_axes=(0, 0, 0, None))
        X = jnp.squeeze(env.X_train)

        bayes_ppds = v_ppd(X, bayes_mean, bayes_cov, obs_noise)
        kf_ppds = v_ppd(X, kf_mean, kf_cov, obs_noise)

        assert jnp.allclose(jnp.squeeze(bayes_ppds[0]),
                            jnp.squeeze(kf_ppds[0]))
        assert jnp.allclose(bayes_ppds[1],
                            kf_ppds[1])'''

if __name__ == '__main__':
    absltest.main()
