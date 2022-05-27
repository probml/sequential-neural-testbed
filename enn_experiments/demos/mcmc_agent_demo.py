import jax.numpy as jnp
from jax import random, vmap

import haiku as hk

import chex
from typing import Dict

import numpy as np
import pandas as pd
import plotnine as gg
from plotnine import ggsave

import enn.base as enn_base
import enn.utils as enn_utils
from enn_experiments.agents.mcmc_agent import MCMCConfig, make_blackjax_agent
from enn_experiments.agents.base import EpistemicSampler, PriorKnowledge

def make_regression_data(key: chex.PRNGKey,
                         n: int,
                         minval: float,
                         maxval:float):
  x_key, y_key = random.split(key)

  x = random.uniform(x_key, shape=(n, 1), minval=minval, maxval=maxval)

  # Define function
  def target_toy(key, x):
      epsilons = random.normal(key, shape=(3,))*0.02
      return (x + 0.3*jnp.sin(2*jnp.pi*(x+epsilons[0])) + 
              0.3*jnp.sin(4*jnp.pi*(x+epsilons[1])) + epsilons[2])

  # Define vectorized version of function
  target_vmap = vmap(target_toy, in_axes=(0, 0), out_axes=0)

  # Generate target values
  keys = random.split(y_key, len(x))
  y = target_vmap(keys, x)
  return x, y

def make_dataset(key: chex.PRNGKey = random.PRNGKey(0),
                 n: int = 100,
                 minval: float = -1.,
                 maxval: float = 1.) -> enn_base.BatchIterator:
  """Factory method to produce an iterator of Batches."""
  x, y = make_regression_data(key,
                              n,
                              minval,
                              maxval)
  data = enn_base.Batch(
      x=x,
      y=y,
  )
  chex.assert_shape(data.x, (None, 1))
  chex.assert_shape(data.y, (None, 1))
  return enn_utils.make_batch_iterator(data)

def make_regression_df(key: chex.PRNGKey = random.PRNGKey(0),
                 n: int = 100,
                 minval: float = -1.,
                 maxval: float = 1.) -> pd.DataFrame:
  """Generate a panda dataframe with sampled predictions."""
  
  x, y =  make_regression_data(key,
                       n,
                       minval,
                       maxval)
  return pd.DataFrame({'x': x[:, 0], 'y': y[:, 0]}).reset_index()


def make_plot_data(sampler: EpistemicSampler,
                   num_sample: int = 20) -> pd.DataFrame:
  """Generate a panda dataframe with sampled predictions."""
  preds_x = np.linspace(0., 0.5).reshape((-1, 1))

  data = []
  rng = hk.PRNGSequence(random.PRNGKey(seed=0))
  for k in range(num_sample):
    preds_y = sampler(preds_x, key=next(rng))
    data.append(pd.DataFrame({'x': preds_x[:, 0], 'y': preds_y[:, 0], 'k': k}))
  plot_df = pd.concat(data)
  return plot_df


def make_plot(sampler: EpistemicSampler,
              num_sample: int = 20,
              dataset_kwargs: Dict = {}) -> gg.ggplot:
  """Generate a regression plot with sampled predictions."""
  
  plot_df = make_plot_data(
      sampler, num_sample=num_sample)

  p = (gg.ggplot()
       + gg.aes('x', 'y')
       + gg.geom_point(data=make_regression_df(**dataset_kwargs), size=3, colour='blue')
       + gg.geom_line(gg.aes(group='k'), data=plot_df, alpha=0.5)
      )
  ggsave(plot=p, filename='mcmc.png', dpi=300)
  return p

def main():
    n = 100
    minval, maxval = 0.0, 0.5
    key = random.PRNGKey(0)
    dataset = make_dataset(key,
                        n,
                        minval,
                        maxval)
    input_dim, tau = 1, 1
    noise_std = 1.
    output_sizes = [16, 16, 1]
    prior = PriorKnowledge(input_dim,
                        n,
                        tau,
                        noise_std=noise_std,
                        output_sizes=output_sizes)
        
    nuts_agent = make_blackjax_agent(MCMCConfig, prior)
    sampler = nuts_agent(dataset)
    
    dataset_kwargs = { "n": n,
                    "minval" : minval,
                    "maxval" : maxval,
                    "key" : random.PRNGKey(0)
                 }
    p = make_plot(sampler,
                dataset_kwargs=dataset_kwargs)
    _ = p.draw()

if __name__ == '__main__':
    main()