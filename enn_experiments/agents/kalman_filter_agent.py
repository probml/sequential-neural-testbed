from enn_experiments.agents.base import EpistemicSampler, PriorKnowledge
import haiku as hk
from jax import random, jit
import jax.numpy as jnp


import chex
import dataclasses
import functools
from typing import Dict, NamedTuple, Optional, Tuple
from acme.utils import loggers

import enn.base as enn_base
import enn.utils as enn_utils
from enn.supervised import base as supervised_base
from enn import networks
from enn import losses

from jsl.lds.kalman_filter import LDS, kalman_filter


class TrainingState(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array


@dataclasses.dataclass
class KalmanFilterConfig:
  """Config Class for Kalman filter."""
  prior_variance: float = 0.1  # Variance of Gaussian prior
  alg_temperature: float = 1  # Temperature parameter for SGLD
  momentum_decay: float = 0.9  # Momentum decay parameter for SGLD
  preconditioner: str = 'None'  # Choice of preconditioner; None or RMSprop
  num_batches: int = 500  # Number of total training steps
  num_warmup: int = 100 # Burn in time for MCMC sampling
  num_samples: int = 100 # Number of MCMC steps per each batch
  seed: int = 0  # Initialization seed
  adaptive_prior_variance: bool = False  # Scale prior_variance with dimension



class KalmanFilterExperiment(supervised_base.BaseExperiment):
  """Class to handle supervised training.
  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps.
  """

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               loss_fn: enn_base.LossFn,
               dataset: enn_base.BatchIterator,
               obs_noise: float = 0.1,
               seed: int = 0,
               logger: Optional[loggers.Logger] = None,
               train_log_freq: int = 1,
               eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None,
               eval_log_freq: int = 1,
               return_history: bool = False):
    self.enn = enn
    self.dataset = dataset
    self.rng = hk.PRNGSequence(seed)
    self.return_history = return_history
    
    # Internalize the loss_fn
    self._loss = jit(functools.partial(loss_fn, self.enn))

    # Internalize the eval datasets
    self._eval_datasets = eval_datasets
    self._eval_log_freq = eval_log_freq
    
    
    # Forward network at random index
    def forward(
        params: hk.Params, inputs: chex.Array, key: chex.PRNGKey) -> chex.Array:
      index = self.enn.indexer(key)
      return self.enn.apply(params, inputs, index)
    self._forward = jit(forward)

    # Define the SGD step on the loss
    def kalman_step(
        state: TrainingState,
        batch: enn_base.Batch,
        key: chex.PRNGKey,
    ) -> Tuple[TrainingState, enn_base.LossMetrics]:
        
        # Calculate the loss, metrics and gradients
      
        x, y = batch.x, batch.y
        
        *_, input_dim = x.shape
        *_, output_dim = y.shape
        
        A = jnp.eye(input_dim)
        Q = 0
        C = lambda t: x[t][None, ...]
        R = obs_noise * jnp.ones((1, 1))
        
        
        lds = LDS(A, C, Q, R,
                 state.mu,
                 state.Sigma)

        mu, Sigma, _, _ = kalman_filter(lds,
                                        y,
                                        return_history=self.return_history)
       
        # metrics.update({'loss': loss})
        if self.return_history:
            mu, Sigma = mu[-1], Sigma[-1]
            return TrainingState(mu, Sigma)
        
        new_state = TrainingState(mu,
                                  Sigma)
        return new_state #TODO: , metrics
    
    self._kalman_step = jit(kalman_step)

    # Initialize networks
    batch = next(self.dataset)
    index = self.enn.indexer(next(self.rng))
    params = self.enn.init(next(self.rng), batch.x, index)
    self.state = TrainingState(*params)
    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'experiment', time_delta=0)
    self._train_log_freq = train_log_freq

  def train(self, num_batches: int):
    """Train the ENN for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      self.state = self._kalman_step(self.state, next(self.dataset), next(self.rng))


  def predict(self, inputs: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Evaluate the trained model at given inputs."""
    return self._forward(self.state.params, inputs, key)

  def loss(self, batch: enn_base.Batch, key: chex.PRNGKey) -> chex.Array:
    """Evaluate the loss for one batch of data."""
    return self._loss(self.state.params, batch, key)


def extract_enn_sampler(enn: enn_base.EpistemicNetwork, 
                        mu: chex.Array,
                        sigma: chex.Array) -> EpistemicSampler:
  """ENN sampler for MCMC."""
  
  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    w = random.multivariate_normal(key, mu, sigma)
    out = enn.apply(w, x, 0)
    return enn_utils.parse_net_output(out)
  return jit(enn_sampler)


def make_kalman_filter_enn(prior: PriorKnowledge) -> enn_base.EpistemicNetwork:
  """Factory method to create fast einsum MLP ensemble ENN.
  This is a specialized implementation for ReLU MLP without a prior network.
  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    num_ensemble: Integer number of elements in the ensemble.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    activation: Jax callable defining activation per layer.
  Returns:
    EpistemicNetwork as an ensemble of MLP.
  """

  # Apply function selects the appropriate index of the ensemble output.
  def apply(params: chex.Array,
            x: chex.Array,
            z: enn_base.Index) -> enn_base.OutputWithPrior:
    del z
    net_out = x @ params
    return net_out

  def init(key: chex.PRNGKey,
           x: chex.Array,
           z: enn_base.Index) -> hk.Params:
    del z
    mu = random.normal(key,  (prior.input_dim, ))
    sigma = jnp.eye(prior.input_dim) * prior.noise_std**2
    return (mu, sigma)

  indexer = networks.indexers.EnsembleIndexer(1)
  
  return enn_base.EpistemicNetwork(apply, init, indexer)


def make_kalman_filter_agent(config: KalmanFilterConfig, prior: PriorKnowledge):
  """Factory method to create a Kalman filter agent."""

  def make_loss(prior) -> enn_base.LossFn:
    
    # L2 loss on perturbed outputs 
    single_loss = losses.L2Loss()
    loss_fn = losses.average_single_index_loss(single_loss, 1)

    # Gaussian prior can be interpreted as a L2-weight decay.
    prior_variance = config.prior_variance
    
    # Scale prior_variance for large input_dim
    if config.adaptive_prior_variance and prior.input_dim >= 100:
      prior_variance *= 2

    scale = (1 / prior_variance) * prior.input_dim / prior.num_train
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
    return loss_fn

  log_freq = int(config.num_batches / 50) or 1

  def kalman_filter_agent(
      dataset: enn_base.BatchIterator
  ) -> EpistemicSampler:

    """Train a MLP via Blackjax Nuts."""
    
    # Define the experiment
    kalman_experiment = KalmanFilterExperiment(
        enn=make_kalman_filter_enn(prior),
        loss_fn=make_loss(prior),
        dataset=dataset, #batch_size=100),
        train_log_freq=log_freq,
    )

    # Train the agent
    step = 0
    for _ in range(config.num_batches):
      step += 1
      kalman_experiment.train(1)

    return extract_enn_sampler(make_kalman_filter_enn(prior), *kalman_experiment.state)

  return kalman_filter_agent