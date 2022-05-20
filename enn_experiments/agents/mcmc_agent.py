from enn_experiments.agents.base import EpistemicSampler, KernelFn, NutsState, PriorKnowledge
import haiku as hk
from jax import random, lax, jit

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import chex
import dataclasses
import functools
from typing import Dict, Optional
from acme.utils import loggers

import enn.base as enn_base
import enn.utils as enn_utils
from enn.supervised import base as supervised_base
from enn import networks
from enn import losses
    
@dataclasses.dataclass
class MCMCConfig:
  """Config Class for MCMC.
  https://github.com/deepmind/neural_testbed/blob/65b90ee36bdcddb044b0e9b3337707f8995c1a1d/neural_testbed/agents/factories/sgmcmc.py#L35
  """
  prior_variance: float = 0.1  # Variance of Gaussian prior
  alg_temperature: float = 1  # Temperature parameter for SGLD
  momentum_decay: float = 0.9  # Momentum decay parameter for SGLD
  preconditioner: str = 'None'  # Choice of preconditioner; None or RMSprop
  num_batches: int = 500  # Number of total training steps
  num_warmup: int = 100 # Burn in time for MCMC sampling
  num_samples: int = 100 # Number of MCMC steps per each batch
  seed: int = 0  # Initialization seed


def inference_loop(rng_key: chex.PRNGKey,
                   kernel: KernelFn,
                   initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = random.split(rng_key, num_samples)
    final, states = lax.scan(one_step, initial_state, keys)

    return final, states


def extract_enn_sampler(enn: enn_base.EpistemicNetwork, 
                        params_list) -> EpistemicSampler:
  """ENN sampler for MCMC."""
  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    param_index = random.randint(key, [], 0, len(params_list))
    fns = [lambda x, w=p: enn.apply(w, x, 0) for p in params_list]
    out = lax.switch(param_index, fns, x)
    return enn_utils.parse_net_output(out)
  return jit(enn_sampler)



class BlackjaxExperiment(supervised_base.BaseExperiment):
  """Class to handle supervised training.
  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps.
  """

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               loss_fn: enn_base.LossFn,
               dataset: enn_base.BatchIterator,
               num_warmup: int,
               num_samples: int,
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
    self._num_warmup = num_warmup
    self._num_samples = num_samples

    # Forward network at random index
    def forward(
        params: hk.Params,
        inputs: enn_base.Array,
        key: enn_base.RngKey) -> enn_base.Array:
      index = self.enn.indexer(key)
      return self.enn.apply(params, inputs, index)
    
    self._forward = jit(forward)

    # Define the step on the loss
    def step(
              state: NutsState,
              batch: enn_base.Batch,
              key: enn_base.RngKey
              ):

        
        @jit
        def partial_logprob(params):
            return -self._loss(params, batch, loss_key)[0]

        warmup_key, sample_key, loss_key = random.split(key, 3)

        state = nuts.new_state(state.position,
                                partial_logprob)

        kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(partial_logprob,
                                                                              step_size,
                                                                              inverse_mass_matrix)
        final, (step_size, inverse_mass_matrix), info = stan_warmup.run(warmup_key,
                                                                        kernel_generator,
                                                                        state,
                                                                        self._num_warmup)

        # Inference
        nuts_kernel = jit(nuts.kernel(partial_logprob,
                                      step_size,
                                      inverse_mass_matrix))

        final, _ = inference_loop(sample_key,
                                        nuts_kernel,
                                        state,
                                        self._num_samples)
                                    
        return final

    # Initialize networks
    batch = next(self.dataset)
    index = self.enn.indexer(next(self.rng))
    params = self.enn.init(next(self.rng), batch.x, index)
    self._step = jit(step)
    self.state = NutsState(params)

    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'experiment', time_delta=0)
    self._train_log_freq = train_log_freq


  def train(self, num_batches: int):

    """Train the ENN for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      self.state = self._step(
          self.state, next(self.dataset), next(self.rng))
      # Periodically log this performance as dataset=train.
      if self.step % self._train_log_freq == 0:
        loss_metrics = {'dataset': 'train',
                        'step': self.step,
                        'sgd': True,
                        'potential_energy' : self.state.potential_energy}
        self.logger.write(loss_metrics)

      # Periodically evaluate the other datasets.
      if self._eval_datasets and self.step % self._eval_log_freq == 0:
        for name, dataset in self._eval_datasets.items():
          loss, metrics = self._loss(
              self.state.params, next(dataset), next(self.rng))
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


def make_blackjax_agent(config: MCMCConfig, prior: PriorKnowledge):
  """Factory method to create a sgmcmc agent."""

  def make_enn(prior) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
        output_sizes=prior.output_sizes,
        num_ensemble=1,
        nonzero_bias=False,
    )

  def make_loss(prior) -> enn_base.LossFn:
    if prior.is_regression:
      # L2 loss on perturbed outputs 
      single_loss = losses.L2Loss()
    else:
      num_classes = config.output_sizes[-1]
      single_loss = losses.combine_single_index_losses_as_metric(
      # This is the loss you are training on.
      train_loss=losses.XentLoss(num_classes),
      # We will also log the accuracy in classification.
      extra_losses={'acc': losses.AccuracyErrorLoss(num_classes)},
      )

    loss_fn = losses.average_single_index_loss(single_loss, 1)

    # Gaussian prior can be interpreted as a L2-weight decay.
    prior_variance = config.prior_variance
    
    # Scale prior_variance for large input_dim
    if config.adaptive_prior_variance and prior.input_dim >= 100:
      prior_variance *= 2

    scale = (1 / prior_variance) * config.input_dim / config.num_train
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
    return loss_fn

  log_freq = int(config.num_batches / 50) or 1

  def blackjax_agent(
      dataset: enn_base.BatchIterator
  ) -> EpistemicSampler:
    """Train a MLP via Blackjax Nuts."""
    # Define the experiment
    blackjax_experiment = BlackjaxExperiment(
        enn=make_enn(prior),
        loss_fn=make_loss(),
        dataset=dataset, #batch_size=100),
        train_log_freq=log_freq,
    )

    # Train the agent
    step = 0
    for _ in range(config.num_batches):
      step += 1
      blackjax_experiment.train(1)
    return extract_enn_sampler(make_enn(prior), params_list)
  
  return blackjax_agent