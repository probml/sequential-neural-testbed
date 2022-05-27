import enn.base as enn_base
from enn import losses
from typing import Any
from enn_experiments.agents.base import PriorKnowledge

DataClass = Any
Prior = PriorKnowledge

def make_loss(config: DataClass,
              prior: PriorKnowledge) -> enn_base.LossFn:
    if prior.is_regression:
      # L2 loss on perturbed outputs 
      single_loss = losses.L2Loss()
    else:
      num_classes = prior.output_sizes[-1]
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

    scale = (1 / prior_variance) * prior.input_dim / prior.num_train
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
    return loss_fn