import jax.numpy as jnp
from jax import random

import chex
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.preprocessing import PolynomialFeatures

import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src.utils import typing as nt_types

import enn.base as enn_base
import enn.utils as enn_utils

from enn_experiments.agents.base import EpistemicSampler, PriorKnowledge
from enn_experiments.agents.kalman_filter_agent import KalmanFilterConfig, make_kalman_filter_agent

def make_dataset(x: chex.Array,
                 y: chex.Array) -> enn_base.BatchIterator:
  """Factory method to produce an iterator of Batches."""
  
  data = enn_base.Batch(
      x=x,
      y=y,
  )
  return enn_utils.make_batch_iterator(data)

def make_regression_df(x: chex.Array,
                       y: chex.Array) -> pd.DataFrame:
  """Generate a panda dataframe with sampled predictions."""
  
  return pd.DataFrame({'x': x[:, 1], 'y': y[:, 0]}).reset_index()


def make_gaussian_sampler(loc: Union[chex.Array, float],
                          scale: Union[chex.Array, float]):
    def gaussian_sampler(key: chex.PRNGKey, shape: Tuple) -> chex.Array:
        return loc + scale * random.normal(key, shape)

    return gaussian_sampler


def make_evenly_spaced_x_sampler(max_val: float,
                                 use_bias: bool = True, min_val: float = 0) -> Callable:
    def eveny_spaced_x_sampler(key: chex.PRNGKey, shape: Tuple) -> chex.Array:
        if len(shape) == 1:
            shape = (shape[0], 1)
        nsamples, nfeatures = shape
        assert nfeatures == 1 or nfeatures == 2

        if nfeatures == 1:
            X = jnp.linspace(min_val, max_val, nsamples)
            if use_bias:
                X = jnp.c_[jnp.ones(nsamples), X]
            else:
                X = X.reshape((-1, 1))
        else:
            step_size = (max_val - min_val) / float(nsamples)
            # define the x and y scale
            x = jnp.arange(min_val, max_val, step_size)
            y = jnp.arange(min_val, max_val, step_size)

            # create all of the lines and rows of the grid
            xx, yy = jnp.meshgrid(x, y)

            # flatten each grid to a vector
            r1, r2 = xx.flatten(), yy.flatten()
            r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
            # horizontal stack vectors to create x1,x2 input for the model
            X = jnp.hstack((r1, r2))
        return X

    return eveny_spaced_x_sampler


def make_linear_kernel(input_dim: int = 1) -> nt_types.AnalyticKernelFn:
    """Generate a linear GP kernel for testing putposes."""
    layers = [
        stax.Dense(1, W_std=1, b_std=1 / np.sqrt(input_dim)),
    ]
    _, _, kernel = stax.serial(*layers)
    return kernel

def make_random_poly_regression_environment(key: chex.PRNGKey,
                                            degree: int,
                                            ntrain: int,
                                            ntest: int,
                                            nout: int = 1,
                                            obs_noise: float = 0.01,
                                            kernel_ridge: float = 1e-6,
                                            x_train_generator: Callable = random.normal,
                                            x_test_generator: Callable = random.normal,
                                            ntk: bool = False,
                                            shuffle: bool = False):
    
    train_key, test_key, y_key, train_noise_key, test_noise_key = random.split(key, 5)

    X_train = x_train_generator(train_key, (ntrain, 1))
    X_test = x_test_generator(test_key, (ntest, 1))

    ntrain = len(X_train)
    ntest = len(X_test)

    X = jnp.vstack([X_train, X_test])

    poly = PolynomialFeatures(degree)
    Phi = jnp.array(poly.fit_transform(X), dtype=jnp.float32)

    N = ntrain + ntest
    get_kernel = 'ntk' if ntk else 'nngp'
    input_dim = X.shape[-1]
    kernel_fn = make_linear_kernel(input_dim)
    kernel = kernel_fn(X, x2=None, get=get_kernel)
    kernel += kernel_ridge * jnp.eye(len(kernel))
    mean = jnp.zeros((N,), dtype=jnp.float32)
    y_function = random.multivariate_normal(y_key, mean, kernel)

    chex.assert_shape(y_function[:ntrain], [ntrain,])

    # Form the training data
    y_noise = random.normal(train_noise_key, [ntrain, 1]) * obs_noise
    y_train = y_function[:ntrain, None] + y_noise

    X_train = Phi[:ntrain]
    X_test = Phi[ntrain:]

    y_noise = random.normal(test_noise_key, [ntest, 1]) * obs_noise
    y_test = y_function[ntrain:, None] + y_noise

    # Form the posterior prediction at cached test data
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn, X_train, y_train, diag_reg=(obs_noise))
    _test_mean, _test_cov = predict_fn(
        t=None, x_test=X_test, get='nngp', compute_cov=True)
    _test_cov += kernel_ridge * jnp.eye(ntest)

    chex.assert_shape(_test_mean, [ntest, 1])
    chex.assert_shape(_test_cov, [ntest, ntest])
    

    if shuffle:
        train_key, test_key = random.split(key)
        train_indices = random.permutation(train_key,
                                           jnp.arange(ntrain))
        test_indices = random.permutation(test_key,
                                          jnp.arange(ntest))

        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        X_test = X_test[test_indices]
    
    return (X_train, y_train), (X_test, y_test)

def make_plot_data(sampler: EpistemicSampler,
                   preds_x: chex.Array,
                   num_sample: int = 20) -> pd.DataFrame:
  """Generate a panda dataframe with sampled predictions."""
  data = []
  keys = random.split(keys, num_sample)
  for k, key in enumerate(keys):
    preds_y = sampler(preds_x, key=key)
    data.append(pd.DataFrame({'x': preds_x[:, 1], 'y': preds_y, 'k': k}))
  plot_df = pd.concat(data)
  return plot_df

def make_plot(sampler,
              training_data: Tuple[chex.Array, chex.Array],
              num_sample: int = 20) -> gg.ggplot:
  """Generate a regression plot with sampled predictions."""
  
  plot_df = make_plot_data(
      sampler, num_sample=num_sample)

  p = (gg.ggplot()
       + gg.aes('x', 'y')
       + gg.geom_point(data=make_regression_df(*training_data), size=3, colour='blue')
       + gg.geom_line(gg.aes(group='k'), data=plot_df, alpha=0.5)
      )

  return p

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
    obs_noise = 0.1


    train_data, unused_test_data = make_random_poly_regression_environment(key,
                                                                        degree,
                                                                        ntrain,
                                                                        ntest,
                                                                        obs_noise=obs_noise,
                                                                        x_train_generator=x_train_generator,
                                                                        x_test_generator=x_test_generator,
                                                                        shuffle=True)
    
    x, y = train_data
    dataset = make_dataset(x, y)

    tau = 1
    noise_std = 0.1
    prior = PriorKnowledge(degree + 1,
                           ntrain,
                           tau,
                           noise_std=noise_std)
    
    kalman_agent = make_kalman_filter_agent(KalmanFilterConfig, prior)
    sampler = kalman_agent(dataset)
    
    preds_x = np.linspace(-0.5, 2).reshape((-1, 1))
    preds_x = PolynomialFeatures(degree).fit_transform(preds_x)


    p = make_plot(sampler, train_data)
    _ = p.draw()
    
if __name__ =="__main__":
    main()
