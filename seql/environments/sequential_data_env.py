import jax.numpy as jnp
from jax import random

import chex
from typing import Callable, Optional


class SequentialDataEnvironment:
    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 true_model: Callable,
                 #y_function: chex.Array,
                 train_batch_size: int,
                 #kernel_ridge: float = 1e-6,
                 tau: int = 1,
                 key: Optional[chex.PRNGKey] = None):
            
        # Checking the dimensionality of our data coming in.
        num_train, input_dim = X_train.shape
        num_test_x_cache, input_dim_test = X_test.shape
        output_dim = y_train.shape[-1]
        assert input_dim == input_dim_test

        self._tau = tau
        self.num_test_x_cache = num_test_x_cache

        self.train_indices = jnp.arange(num_train)
        self.test_indices = jnp.arange(num_test_x_cache)

        # TODO: It will produce an error if ntrain % train_batch_size != 0
        ntrain_batches = num_train // train_batch_size
        
        self.X_train = jnp.reshape(X_train, [ntrain_batches, train_batch_size, input_dim])
        self.y_train = jnp.reshape(y_train, [ntrain_batches, train_batch_size, output_dim])
        self.X_test = X_test
        
        if key is not None:
            self.shuffle_data(key)
        self.true_model = true_model

    def get_data(self, t: int):
        return self.X_train[t], self.y_train[t]

    def shuffle_data(self, key: chex.PRNGKey):
        ntrain_batches, train_batch_size, input_dim = self.X_train.shape
        output_dim = self.y_train.shape[-1]

        self.X_train = jnp.reshape(self.X_train, [-1, input_dim])
        self.y_train = jnp.reshape(self.y_train, [-1, output_dim])

        train_key, key = random.split(key)

        self.train_indices = random.permutation(train_key, self.train_indices)
        self.X_train = self.X_train[self.train_indices]
        self.y_train = self.y_train[self.train_indices]

        self.X_train = jnp.reshape(self.X_train, [ntrain_batches, train_batch_size, input_dim])
        self.y_train = jnp.reshape(self.y_train, [ntrain_batches, train_batch_size, output_dim])


    def reset(self, key: chex.PRNGKey):
        self.shuffle_data(key)
