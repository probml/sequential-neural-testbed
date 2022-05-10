import jax.numpy as jnp
from jax import nn, vmap

import chex
from typing import Callable, Tuple

from seql.utils import categorical_log_likelihood
 
# Maps logits=[a, b, c], labels=[b, 1] to float metric.
# Where a=num_enn_samples, b=batch_size, c=num_classes.
MetricsCalculator = Callable[[chex.Array, chex.Array], float]

 
def safe_average(x: chex.Array) -> float:
    max_val = jnp.max(x)
    return jnp.log(jnp.mean(jnp.exp(x - max_val))) + max_val

def reshape_to_smaller_batches(
    logits: chex.Array,
    labels: chex.Array,
    batch_size: int) -> Tuple[chex.Array, chex.Array]:
  """Reshapes logits,labels to add leading batch_size dimension.
 
  In case the size of logits and labels are such that they cannot be equally
  divided into batches of size batch_size, extra data is discarded.
 
  Args:
    logits: has shape [num_enn_samples, num_data, num_classes]
    labels: has shape [num_data, 1]
    batch_size: desired output batch size.
 
  Returns:
    A tuple of batched_logits and batched_labels with shapes
      batched_logits: (num_batches, num_enn_samples, batch_size, num_classes)
      batched_labels: (num_batches, batch_size, 1)
  """
  # Shape checking
  assert len(logits.shape) == 3
  num_enn_samples, num_data, num_classes = logits.shape
  chex.assert_shape(labels, [num_data, 1])
  assert num_data >= batch_size

  ##############################################################################
  # 1. We split num_data to batches of size batch_size. To ensure that the split
  # is possible, we might need to discard extra data.
  num_batches = num_data // batch_size
  num_extra_data = num_data % batch_size
  num_data -= num_extra_data
 
  # 1.1. Discard extra data if needed.
  logits = logits[:, :num_data, :]
  labels = labels[:num_data, :]
  chex.assert_shape(logits, [num_enn_samples, num_data, num_classes])
  chex.assert_shape(labels, [num_data, 1])
 
  # 1.2. Split num_data to batches of size batch_size
  batched_logits = jnp.reshape(
      logits, [num_enn_samples, num_batches, batch_size, num_classes])
  batched_labels = jnp.reshape(labels, [num_batches, batch_size, 1])
 
  ##############################################################################
  # 2. We want num_batches to be the leading axis. It is already the case for
  # batched_labels, but we need to change axes for batched_logits.
  batched_logits = jnp.transpose(batched_logits, [1, 0, 2, 3])
  chex.assert_shape(batched_logits,
                  [num_batches, num_enn_samples, batch_size, num_classes])
 
  return batched_logits, batched_labels



def make_nll_polyadic_calculator(
    tau: int = 10, kappa: int = 2) -> MetricsCalculator:
  """Returns a MetricCalculator that computes d_{KL}^{tau, kappa} metric."""
  assert tau % kappa == 0
 
  def joint_ll_repeated(logits:chex.Array, labels:chex.Array) -> float:
    """Calculates joint NLL evaluated on anchor points repeated tau / kappa."""
    # Shape checking
    chex.assert_shape(logits, [kappa, None])
    chex.assert_shape(labels, [kappa, 1])
 
    # Compute log-likehood, and then multiply by tau / kappa repeats.
    probs = nn.softmax(logits, axis=-1)
    ll = categorical_log_likelihood(probs, labels)
    num_repeat = tau / kappa
    return ll * num_repeat
 
  def enn_nll(logits:chex.Array, labels:chex.Array) -> float:
    """Averages NLL over multiple ENN samples."""
    # Shape checking
    chex.assert_shape(logits, [None, kappa, None])
    chex.assert_shape(labels, [kappa, 1])
    batched_labels = jnp.repeat(labels[None], logits.shape[0], axis=0)
 
    # Averaging over ENN samples
    lls = vmap(joint_ll_repeated, in_axes=(0, 0))(logits, batched_labels)
    return -1 * safe_average(lls)
 
  def polyadic_nll(logits: chex.Array, labels: chex.Array) -> float:
    """Returns polyadic NLL based on repeated inputs.
 
    Internally this function works by taking the batch of logits and then
    "melting" it to add an extra dimension so that the batches we evaluate
    likelihood are of size=kappa. This means that one batch_size=N*kappa becomes
    N batches of size=kappa, each of which we evaluate the NLL as if the data
    had been repeated tau/kappa times. This does *NOT* exactly match the
    polyadic sampling scheme described in the paper, but should be similar.
    We use this reshape strategy to optimize memory/compute usage for now.
 
    Args:
      logits: [num_enn_samples, batch_size, num_classes]
      labels: [batch_size, 1]
    """
    # TODO(iosband): Revisit metric/performance and sampling solution.
    # Shape checking
    chex.assert_shape(labels, [logits.shape[1], 1])
 
    # Creating synthetic batches of size=kappa then use vmap.
    batched_logits, batched_labels = reshape_to_smaller_batches(
        logits, labels, batch_size=kappa)
    nlls = vmap(enn_nll, in_axes=(0, 0))(batched_logits, batched_labels)
    return jnp.mean(nlls)
 
  return polyadic_nll

