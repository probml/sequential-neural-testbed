from typing import Callable, Optional
import jax.numpy as jnp
from jax import random, vmap

import chex
from typing import Tuple

from seql.environments.sequential_data_env import SequentialDataEnvironment
from seql.utils import categorical_log_likelihood


class SequentialClassificationEnvironment(SequentialDataEnvironment):

    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 true_model: Callable,
                 train_batch_size: int,
                 logprobs: chex.Array,
                 key: Optional[chex.PRNGKey] = None):
        ntrain = len(y_train)
        _, out = logprobs.shape
        ntrain_batches = ntrain // train_batch_size

        self.train_logprobs = jnp.reshape(logprobs[:ntrain], [ntrain_batches, train_batch_size, out])

        super().__init__(X_train,
                         y_train,
                         X_test,
                         true_model,
                         train_batch_size,
                         key)

    def shuffle_data(self, key: chex.PRNGKey):
        super().shuffle_data(key)

        ntrain_batches, train_batch_size, out = self.y_train.shape
        self.train_logprobs = jnp.reshape(self.train_logprobs, [-1, out])
        self.train_logprobs = self.train_logprobs[self.train_indices]
        self.train_logprobs = jnp.reshape(self.train_logprobs, [ntrain_batches, train_batch_size, out])