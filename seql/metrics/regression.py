import jax.numpy as jnp

import chex
import distrax

from seql.metrics.utils import average_sampled_log_likelihood

def gaussian_log_likelihood(mu: chex.Array,
                            cov: chex.Array,
                            predictions) -> float:
    return jnp.sum(distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1), cov).log_prob(predictions))


def gaussian_sample_kl(
                sampled_ll: chex.Array,
                true_ll: chex.Array) -> float:
    """Computes KL estimate on a single instance of test data."""
    ave_ll = average_sampled_log_likelihood(sampled_ll)
    ave_true_ll = average_sampled_log_likelihood(true_ll)
    return ave_true_ll - ave_ll