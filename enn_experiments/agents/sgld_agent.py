from enn_experiments.agents.base import EpistemicSampler, IntegratorState, PriorKnowledge, SGLDState
import haiku as hk
from jax import random, jit, tree_flatten, tree_map, vmap
import jax.numpy as jnp

import chex
import dataclasses
import functools
from typing import Callable, Dict, Optional
from acme.utils import loggers

from sgmcmcjax.samplers import build_sgld_sampler

import enn.base as enn_base
import enn.utils as enn_utils
from enn.supervised import base as supervised_base
from enn import networks
from enn import losses

from enn_experiments.agents.utils import make_loss
    
@dataclasses.dataclass
class SGLDConfig:
  """Config Class for SGLD.
  https://github.com/deepmind/neural_testbed/blob/65b90ee36bdcddb044b0e9b3337707f8995c1a1d/neural_testbed/agents/factories/sgmcmc.py#L35
  """
  dt: float
  batch_size: int
  alpha: float = 0.99
  eps: float = 1e-5
  prior_variance: float = 0.1  # Variance of Gaussian prior
  num_batches: int = 1  # Number of total training steps
  num_samples: int = 800 # Number of SGLD steps per each batch
  seed: int = 0  # Initialization seed    
  adaptive_prior_variance: bool = False  # Scale prior_variance with dimension


def extract_enn_sampler(enn: enn_base.EpistemicNetwork, 
                        params_list) -> EpistemicSampler:
    """ENN sampler for SGLD."""
  
    if isinstance(params_list, (jnp.ndarray, jnp.generic, list)):
        num_params = len(params_list)
    else:
        params, unflatten_fn = tree_flatten(params_list)
        num_params = len(params[0])
        
    def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
        """Generate a random sample from posterior distribution at x."""
        param_index = random.randint(key, [], 0, num_params)
        outs = vmap(lambda w, x, z: enn.apply(w, x, z),
                    in_axes=(0, None, None))(params_list,  x, 0)
        out = outs[param_index]
        return enn_utils.parse_net_output(out)

    return jit(enn_sampler)



class SgmcmcjaxExperiment(supervised_base.BaseExperiment):
  """Class to handle supervised training.
  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps.
  """

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               loss_fn: enn_base.LossFn,
               dataset: enn_base.BatchIterator,
               num_samples: int,
               dt: float,
               batch_size: int,
               seed: int = 0,
               logger: Optional[loggers.Logger] = None,
               train_log_freq: int = 1,
               eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None,
               eval_log_freq: int = 1):
    
    self.enn = enn
    self.dataset = dataset
    self.rng = hk.PRNGSequence(seed)

    # Internalize the loss_fn
    self._loss = jit(functools.partial(loss_fn, self.enn))

    # Internalize the eval datasets
    self._eval_datasets = eval_datasets
    self._eval_log_freq = eval_log_freq
    self._num_samples = num_samples

    # Forward network at random index
    def forward(
        params: hk.Params,
        inputs: enn_base.Array,
        key: enn_base.RngKey) -> enn_base.Array:
      index = self.enn.indexer(key)
      return self.enn.apply(params, inputs, index)
    
    self._forward = jit(forward)
    
    def loglikelihood(params, x, y):
      batch = enn_base.Batch(x.reshape((-1, x.shape[-1])),
                             y.reshape((-1, y.shape[-1])),
                             jnp.array([[0]]))
      
      unused_key = random.PRNGKey(0)
      return -self._loss(params, batch, unused_key)[0]
    
    logprior = lambda params: 0.
  
    # Define the step on the loss
    def step(
              state: SGLDState,
              batch: enn_base.Batch,
              key: enn_base.RngKey
              ):
      
      sampler = build_sgld_sampler(dt,
                                   loglikelihood,
                                   logprior,
                                   (batch.x, batch.y),
                                   batch_size)
      samples = sampler(key, num_samples, state.params)
      params = tree_map(lambda x:x[-1], samples)
      return SGLDState(params, samples)

    # Initialize networks
    batch = next(self.dataset)
    index = self.enn.indexer(next(self.rng))
    params = self.enn.init(next(self.rng), batch.x, index)
    self.state = SGLDState(params)
    
    self._step = jit(step)

    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'experiment', time_delta=0)
    self._train_log_freq = train_log_freq


  def train(self, num_batches: int):

    """Train the ENN for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      
      self.state = self._step(self.state, next(self.dataset), next(self.rng))
      
      # Periodically log this performance as dataset=train.
      if self.step % self._train_log_freq == 0:
        loss_metrics = {'dataset': 'train',
                        'step': self.step,
                        'sgd': True}
                        #'potential_energy' : self.state.final.potential_energy}
        self.logger.write(loss_metrics)

      # Periodically evaluate the other datasets.
      if self._eval_datasets and self.step % self._eval_log_freq == 0:
        for name, dataset in self._eval_datasets.items():
          loss, metrics = self._loss(
              self.state.final.params, next(dataset), next(self.rng))
          metrics.update({
              'dataset': name,
              'step': self.step,
              'sgd': False,
              'loss': loss,
          })
          self.logger.write(metrics)

  def predict(self, inputs: enn_base.Array, key: enn_base.RngKey) -> enn_base.Array:
    """Evaluate the trained model at given inputs."""
    return self._forward(self.state.params, inputs, key)

  def loss(self, batch: enn_base.Batch, key: enn_base.RngKey) -> enn_base.Array:
    """Evaluate the loss for one batch of data."""
    return self._loss(self.state.params, batch, key)


def make_sgmcmcjax_agent(config: SGLDConfig, prior: PriorKnowledge):
  """Factory method to create a sgmcmc agent."""
  
  def make_enn(prior) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
        output_sizes=prior.output_sizes,
        num_ensemble=1)

  log_freq = int(config.num_batches / 50) or 1

  def sgmcmcjax_agent(
      dataset: enn_base.BatchIterator
  ) -> EpistemicSampler:
    """Train a MLP via SgmcmcJax SGLD."""
    # Define the experiment
    sgmcmcjax_experiment = SgmcmcjaxExperiment(
        enn=make_enn(prior),
        loss_fn=make_loss(config, prior),
        dataset=dataset,
        dt=config.dt,
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        train_log_freq=log_freq,
    )

    # Train the agent
    sgmcmcjax_experiment.train(500)
    return extract_enn_sampler(make_enn(prior), sgmcmcjax_experiment.state.samples)
  
  return sgmcmcjax_agent