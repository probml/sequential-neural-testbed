import dataclasses
from typing import Any, Dict, Optional, Tuple
import typing_extensions

import chex

Data = Tuple[chex.Array, chex.Array]

@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
  """What an agent knows a priori about the problem."""
  input_dim: int
  num_train: int
  tau: int
  num_classes: int = 1
  layers: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = None
  extra: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class ENNQuality:
  kl_estimate: float
  extra: Optional[Dict[str, Any]] = None


class EpistemicSampler(typing_extensions.Protocol):
  """Interface for drawing posterior samples from distribution.
  For classification this should represent the class *logits*.
  For regression this is the posterior sample of the function f(x).
  Assumes a batched input x.
  """

  def __call__(self, key: chex.PRNGKey, x: chex.Array) -> chex.Array:
    """Generate a random sample from approximate posterior distribution."""