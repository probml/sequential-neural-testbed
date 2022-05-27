import dataclasses
import chex
import typing_extensions
from typing import Any, Dict, NamedTuple, Optional, Tuple

State = NamedTuple
Info = NamedTuple


@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
    """What an agent knows a priori about the problem.
    https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/base.py#L33
    """
    input_dim: int
    num_train: int
    tau: int
    output_dim: int = 1
    output_sizes: Optional[int] = None
    noise_std: Optional[float] = None
    temperature: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None
    is_regression: bool = True

class IntegratorState(NamedTuple):
    """State of the trajectory integration.
    We keep the gradient of the potential energy to speedup computations.
    # https://github.com/blackjax-devs/blackjax/blob/fd83abf6ce16f2c420c76772ff2623a7ee6b1fe5/blackjax/mcmc/integrators.py#L12
    """
    position: chex.ArrayTree
    momentum: chex.ArrayTree = None
    potential_energy: float = None
    potential_energy_grad: chex.ArrayTree = None
    
class NutsState(NamedTuple):
    final: IntegratorState
    samples: Optional[chex.ArrayTree] = None
    
class SGLDState(NamedTuple):
    params: chex.ArrayTree
    samples: Optional[chex.ArrayTree] = None

class KernelFn(typing_extensions.Protocol):
    """A transition kernel used as the `update` of a `SamplingAlgorithms`.
        https://github.com/blackjax-devs/blackjax/blob/main/blackjax/base.py#L40
    """

    def __call__(self, rng_key: chex.PRNGKey,
                state: State) -> Tuple[State, Info]:
        ...

class EpistemicSampler(typing_extensions.Protocol):
    """Interface for drawing posterior samples from distribution.
    For classification this should represent the class *logits*.
    For regression this is the posterior sample of the function f(x).
    Assumes a batched input x.
    https://github.com/deepmind/neural_testbed/blob/65b90ee36bdcddb044b0e9b3337707f8995c1a1d/neural_testbed/base.py#L51
    """

    def __call__(self, x: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Generate a random sample from approximate posterior distribution."""