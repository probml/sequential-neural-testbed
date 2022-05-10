# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/leaderboard/plotting.py
from functools import reduce
import jax.numpy as jnp
from jax import nn, random

import haiku as hk

import numpy as np

import seaborn as sns
import plotnine as gg
import pandas as pd

import chex
from typing import Dict
from seql.agents.base import Agent, BeliefState
from seql.environments.sequential_data_env import SequentialDataEnvironment

from seql.experiments.base import PriorKnowledge


agents = {
          "kf": 0,
          "eekf": 0,
          "exact bayes": 1,
          "sgd": 2,
          "laplace": 3,
          "bfgs": 4,
          "lbfgs": 5,
          "nuts": 6,
          "sgld": 7,
          "scikit": 8,
          "ensemble":9
          }

colors = {k: sns.color_palette("Paired")[v]
          for k, v in agents.items()}


def sort_data(x, y):
    *_, nfeatures = x.shape
    *_, ntargets = y.shape

    x_ = x.reshape((-1, nfeatures))
    y_ = y.reshape((-1, ntargets))

    if nfeatures > 1:
        indices = jnp.argsort(x_[:, 1])
    else:
        indices = jnp.argsort(x_[:, 0])

    x_ = x_[indices]
    y_ = y_[indices]

    return x_, y_, indices


def plot_regression_posterior_predictive(ax,
                                         X,
                                         y,
                                         X_line,
                                         posterior_predictive_outputs,
                                         agent_name,
                                         t):
    nprev = reduce(lambda x, y: x * y,
                   X[:t].shape[:-1])

    X_test, y_test = X[:t+1], y[:t+1]

    nfeatures = X_test.shape[-1]
    prev_x = X_test.reshape((-1, nfeatures))[:nprev]
    prev_y = y_test.reshape((-1, 1))[:nprev]

    cur_x = X_test.reshape((-1, nfeatures))[nprev:]
    cur_y = y_test.reshape((-1, 1))[nprev:]

    # Plot training data
    ax.scatter(prev_x[:, 1],
               prev_y,
               c="#40476D")
    
    ax.scatter(cur_x[:, 1],
               cur_y,
               c="#c33149",
               marker="^")

    X_test, y_test, _ = sort_data(cur_x, cur_y)

    ypred, error = posterior_predictive_outputs

    ypred = jnp.squeeze(ypred)
    print(X_line.shape)
    error = jnp.squeeze(error)
    '''ax.plot(X_test,
            ground_truth,
            color="k")
      '''
    
    color = colors[agent_name]

    ax.plot(X_line,
            ypred,
            color=color)

    ax.fill_between(X_line,
                    ypred + error,
                    ypred - error,
                    alpha=0.2,
                    color=color)


def plot_classification_2d(ax,
                           env,
                           grid,
                           grid_preds,
                           t):
    sns.set_style("whitegrid")
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

    x, y = sort_data(env.X_test[:t + 1], env.y_test[:t + 1])
    nclasses = y.max()

    if nclasses == 1 and grid_preds.shape[-1] == 1:
        grid_preds = jnp.hstack([1 - grid_preds, grid_preds])

    '''ax.contourf(grid[:, 1].reshape((100, 100)),
                grid[:, 2].reshape((100, 100)),
                grid_preds[:, 1].reshape((100,100)),
                cmap=cmap)'''

    for cls in range(nclasses + 1):
        indices = jnp.argwhere(y == cls)

        # Plot training data
        ax.scatter(x[indices, 1],
                   x[indices, 2])

############################################################
# Specialized plots for 2D problems
BLUE = '#084594'
RED = '#e41a1c'


def set_gg_theme():
  """Sets the global ggplot theme."""
  try:
    # TODO(author2): Understand why this is causing errors in testing.
    gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
    gg.theme_update(figure_size=(12, 8), panel_spacing=0.5)
  except RuntimeError:
    pass


def gen_2d_grid(plot_range: float) -> np.ndarray:
  """Generates a 2D grid for data in a certain_range."""
  data = []
  x_range = np.linspace(-plot_range, plot_range)
  for x1 in x_range:
    for x2 in x_range:
      data.append((x1, x2))
  return np.vstack(data)


def _gen_samples_2d(agent: Agent,
                    belief: BeliefState,
                    x: chex.Array,
                    num_samples: int,
                    categorical: bool = False) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  # Generate the samples
  data = []
  rng = hk.PRNGSequence(random.PRNGKey(seed=0))
  for seed in range(num_samples):
    # IT'S CHANGED!
    theta = agent.sample_params(next(rng), belief)
    net_out = agent.model_fn(theta, x)
    #########
    y = nn.softmax(net_out, axis=-1)[:, 1] if categorical else net_out[:, 0]
    df = pd.DataFrame({'x0': x[:, 0], 'x1': x[:, 1], 'y': y, 'seed': seed})
    data.append(df)
  return pd.concat(data)

def _agg_samples_2d(sample_df: pd.DataFrame) -> pd.DataFrame:
  """Aggregate ENN samples for plotting."""
  def pct_95(x):
    return np.percentile(x, 95)
  def pct_5(x):
    return np.percentile(x, 5)
  enn_df = (sample_df.groupby(['x0', 'x1'])['y']
            .agg([np.mean, np.std, pct_5, pct_95]).reset_index())
  enn_df = enn_df.rename({'mean': 'y'}, axis=1)
  enn_df['method'] = 'enn'
  return enn_df

def _plot_expanded_2d(problem_df: pd.DataFrame,
                      enn_df: pd.DataFrame,
                      train_df: pd.DataFrame) -> gg.ggplot:
  """Side-by-side plot comparing ENN and true function with pct_5, pct_95."""
  plt_df = pd.melt(enn_df, id_vars=['x0', 'x1'],
                   value_vars=['y', 'pct_5', 'pct_95'])
  plt_df['variable'] = plt_df.variable.apply(lambda x: 'enn:' + x)
  problem_df['value'] = problem_df['y']
  problem_df['variable'] = 'true_function'
  p = (gg.ggplot(pd.concat([problem_df, plt_df]))
       + gg.aes(x='x0', y='x1', fill='value')
       + gg.geom_tile()
       + gg.geom_point(gg.aes(fill='y'), data=train_df, size=3, stroke=1.5,
                       alpha=0.7)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('variable')
       + gg.theme(figure_size=(12, 10))
       + gg.ggtitle('Comparing ENN and true probabilities'))
  return p

def _gen_problem_2d(
    problem,
    x: chex.Array,
) -> pd.DataFrame:
  """Generate underlying problem dataset."""
  assert x.shape[1] == 2
  # IT'S CHANGED!
  logits = problem.true_model(x)  # pylint:disable=protected-access
  test_probs = nn.softmax(logits, axis=-1)[:, 1]
  np_data = np.hstack([x, test_probs[:, None]])
  problem_df = pd.DataFrame(np_data, columns=['x0', 'x1', 'y'])
  problem_df['method'] = 'true_function'
  return problem_df


def _make_train_2d(
    problem):
  # IT'S CHANGED!
  x, y = problem.X_train, problem.y_train
  x = x.reshape((-1, x.shape[-1]))
  y = y.reshape((-1, y.shape[-1]))
  #### 
  return pd.DataFrame(np.hstack([x, y]), columns=['x0', 'x1', 'y'])

def _plot_error_2d(problem_df: pd.DataFrame, enn_df: pd.DataFrame,
                   train_df: pd.DataFrame) -> gg.ggplot:
  """Single plot of error in ENN."""
  plt_df = pd.merge(
      enn_df, problem_df, on=['x0', 'x1'], suffixes=('_enn', '_problem'))
  p = (gg.ggplot(plt_df)
       + gg.aes(x='x0', y='x1', fill='y_problem - y_enn')
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0)
       + gg.geom_tile()
       + gg.geom_point(gg.aes(x='x0', y='x1', fill='y'), data=train_df, size=3,
                       stroke=1.5, inherit_aes=False, show_legend=False)
       + gg.theme(figure_size=(7, 5))
       + gg.ggtitle('Error in ENN mean estimation')
      )
  return p


def _plot_std_2d(enn_df: pd.DataFrame,
                 train_df: pd.DataFrame) -> gg.ggplot:
  """Single plot of standard deviation in ENN predications."""
  p = (gg.ggplot(enn_df)
       + gg.aes(x='x0', y='x1', fill='std')
       + gg.scale_fill_gradient2('white', '#005a32', '#ffff33', midpoint=0.1)
       + gg.geom_tile()
       + gg.geom_point(gg.aes(x='x0', y='x1', colour='y'), data=train_df,
                       size=3, inherit_aes=False, show_legend=False, alpha=0.7)
       + gg.scale_colour_gradient(BLUE, RED, limits=[0, 1])
       + gg.theme(figure_size=(7, 5))
       + gg.ggtitle('Standard deviation in ENN predications')
       )
  return p

def _plot_default_2d(problem_df: pd.DataFrame,
                     enn_df: pd.DataFrame,
                     train_df: pd.DataFrame) -> gg.ggplot:
  """Side-by-side plot comparing ENN and true function."""
  p = (gg.ggplot(pd.concat([problem_df, enn_df]))
       + gg.aes(x='x0', y='x1', fill='y')
       + gg.geom_tile()
       + gg.geom_point(data=train_df, size=3, stroke=1.5, alpha=0.7)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('method')
       + gg.theme(figure_size=(12, 5))
       + gg.ggtitle('Comparing ENN and true probabilities')
      )
  return p
  
def _plot_enn_samples_2d(sample_df: pd.DataFrame,
                         train_df: pd.DataFrame) -> gg.ggplot:
  """Plot realizations of enn samples."""
  p = (gg.ggplot(sample_df)
       + gg.aes(x='x0', y='x1', fill='y')
       + gg.geom_tile()
       + gg.geom_point(data=train_df, size=3, stroke=1.5)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('seed', labeller='label_both')
       + gg.theme(figure_size=(18, 12), panel_spacing=0.1)
       + gg.ggtitle('ENN sample realizations')
      )
  return p

def generate_2d_plots(
    true_model,
    agent: Agent,
    belief: BeliefState,
    num_samples: int = 20) -> Dict[str, gg.ggplot]:
  """Generates a sequence of plots for debugging."""
  x = gen_2d_grid(3)
  sample_df = _gen_samples_2d(agent, belief, x, num_samples, categorical=True)
  enn_df = _agg_samples_2d(sample_df)
  problem_df = _gen_problem_2d(true_model, x)
  train_df = _make_train_2d(true_model)
  return {
      'enn': _plot_default_2d(problem_df, enn_df, train_df),
      'more_enn': _plot_expanded_2d(problem_df, enn_df, train_df),
      'err_enn': _plot_error_2d(problem_df, enn_df, train_df),
      'std_enn': _plot_std_2d(enn_df, train_df),
      'sample_enn': _plot_enn_samples_2d(sample_df, train_df),
  }


def _gen_samples(agent: Agent,
                 belief: BeliefState,
                 x: chex.Array,
                 num_samples: int,
                 categorical: bool = False) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  # Generate the samples
  data = []
  rng = hk.PRNGSequence(random.PRNGKey(seed=0))
  for seed in range(num_samples):
    # IT'S CHANGED!
    theta = agent.sample_params(next(rng), belief)
    net_out = agent.model_fn(theta, x)
    #########
    y = nn.softmax(net_out)[:, 1] if categorical else net_out[:, 0]
    data.append(pd.DataFrame({'x': x[:, 0], 'y': y, 'seed': seed}))
  sample_df = pd.concat(data)

  # Aggregate the samples for plotting
  def pct_95(x):
    return np.percentile(x, 95)
  def pct_5(x):
    return np.percentile(x, 5)
  enn_df = (sample_df.groupby('x')['y']
            .agg([np.mean, np.std, pct_5, pct_95]).reset_index())
  enn_df = enn_df.rename({'mean': 'y'}, axis=1)
  enn_df['method'] = 'enn'
  return enn_df

def plot_1d_regression(environment: SequentialDataEnvironment,
                       agent: Agent,
                       belief: BeliefState,
                       num_samples: int = 100) -> gg.ggplot:
  """Plots 1D regression with confidence intervals."""
  # Training data
  #train_data = gp_model.train_data
  x = np.array(environment.X_train.reshape((-1, environment.X_train.shape[-1]))[:, 0])
  y = np.array(environment.y_train.reshape((-1, 1))[:, 0])
  
  df = pd.DataFrame({'x': x, 'y': y})

  # Posterior data
  posterior_df = pd.DataFrame({
      'x': np.array(environment.X_test[:, 0]),
      'y': np.array(environment.test_mean[:, 0]),
      'std': np.sqrt(np.diag(environment.test_cov)),
  })
  posterior_df['method'] = 'gp'
  # ENN data
  enn_df = _gen_samples(agent, belief, environment.X_test, num_samples)
  p = (gg.ggplot(pd.concat([posterior_df, enn_df]))
       + gg.aes(x='x', y='y', ymin='y-std', ymax='y+std', group='method')
       + gg.geom_ribbon(gg.aes(fill='method'), alpha=0.25)
       + gg.geom_line(gg.aes(colour='method'), size=2)
       + gg.geom_point(gg.aes(x='x', y='y'), data=df, size=4, inherit_aes=False)
       + gg.scale_colour_manual(['#e41a1c', '#377eb8'])
       + gg.scale_fill_manual(['#e41a1c', '#377eb8'])
      )
  return p

def sanity_1d(environment: SequentialDataEnvironment,
              agent: Agent,
              belief: BeliefState) -> gg.ggplot:
  """Sanity check to plot 1D representation of the GP testbed output."""
  set_gg_theme()
  #if not isinstance(gp_model, gp_regression.GPRegression):
  #print('WARNING: no plot implemented')
  #return gg.ggplot()
  return plot_1d_regression(environment, agent, belief)
  '''else:
    if not isinstance(true_model, likelihood.SampleBasedTestbed):
      raise ValueError('Unrecognised testbed for classification plot.')
    return plot_1d_classification(true_model, enn_sampler)'''

def sanity_plots(
    prior: PriorKnowledge,
    environment: SequentialDataEnvironment,
    agent: Agent,
    belief: BeliefState
) -> Dict[str, gg.ggplot]:
  """Sanity check plots for output of GP testbed output."""
  set_gg_theme()
  # Specialized plotting for the 2D classification infra.
  if prior.num_classes == 2 and prior.input_dim == 2:
    # TODO(author2): annotate true_model as classification.
    return generate_2d_plots(environment, agent, belief)  # pytype:disable=wrong-arg-types
  else:
    return {'enn': sanity_1d(environment, agent, belief)}