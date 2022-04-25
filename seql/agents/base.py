import jax.numpy as jnp
from jax import nn, vmap, random
from jax.scipy.special import logsumexp

import distrax

import chex
import typing_extensions
from typing import NamedTuple, Tuple, Callable



AgentInitFn = Callable
BeliefState = NamedTuple
Info = NamedTuple
Params = chex.ArrayTree
SampleFn = Callable
ModelFn = Callable


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: chex.Array,
                 x: chex.Array):
        ...

class LoglikelihoodFn(typing_extensions.Protocol):

    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array,
                 model_fn: ModelFn):
        ...


class LogpriorFn(typing_extensions.Protocol):

    def __call__(self,
                 params: Params):
        ...

class Agent:
    '''
    Agent interface.
    '''

    def __init__(self,
                 is_classifier: bool):
        
        self.is_classifier = is_classifier

        if is_classifier:
            self.predictive_distribution_given_params = self.predict_given_params_classification
        else:
            self.predictive_distribution_given_params = self.predict_given_params_regression

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array) -> Tuple[BeliefState, Info]:
        '''
        Update the belief state given the training data.
        '''
        pass

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState) -> chex.Array:
        '''
        Sample parameters given the belief state.
        '''
        pass

    def predict_given_params_regression(self,
                                        params: chex.ArrayTree,
                                        x: chex.Array):
        # Regression with C outputs (independent)

        mu = self.model_fn(params, x)

        # n test examples, dimensionality of output
        ntest, output_dim = jnp.shape(mu)
        pred_dist = distrax.MultivariateNormalDiag(mu,
                                                   self.obs_noise * jnp.ones((ntest, output_dim)))

        return pred_dist

    def predict_given_params_classification(self,
                                            params: chex.ArrayTree,
                                            x: chex.Array):
        # Classifier with C outputs (one hot)
        logits = self.model_fn(params, x)
        probs = nn.softmax(logits, axis=-1)  # normalize over C
        pred_dist = distrax.Categorical(probs=probs)
        return pred_dist

    def posterior_predictive_sample(self,
                                    key: chex.PRNGKey,
                                    belief: BeliefState,
                                    x: chex.Array,
                                    nsamples_params: int,
                                    nsamples_output: int) -> chex.Array:

        # x : batch size * number of features(N * D)
        # Y : number of sampled params * batch size * number of sampled outputs * output dimensionality 
        # Y : S * N * M * C where M = nsamples_outout, C  = |Y|

        def sample_from_predictive_distribution(key, theta, x):
            inputs = x.reshape((1, -1))
            pred_dist = self.predictive_distribution_given_params(theta, inputs)
            samples = pred_dist.sample(seed=key,
                                       sample_shape=(nsamples_output,))
            samples = samples.reshape((nsamples_output, -1))
            return samples

        def sample_from_belief(key):
            param_key, predictive_key = random.split(key, 2)
            theta = self.sample_params(param_key, belief)
            nsamples_input = len(x)
            keys = random.split(predictive_key, nsamples_input)
            vsample = vmap(sample_from_predictive_distribution,
                           in_axes=(0, None, 0))
            samples = vsample(keys, theta, x)
            return samples

        keys = random.split(key, nsamples_params)
        return vmap(sample_from_belief)(keys)

    def logprob_given_belief(self,
                             key: chex.PRNGKey,
                             belief: BeliefState,
                             X: chex.Array,
                             Y: chex.Array,
                             nsamples_params: int) -> chex.Array:
        # X: N*D
        # Y: N*C
        # returns N*1 
        # (N=batch size, C=event shape)
        # P(s,n) = p(y(n,:) | x(n,:), theta(s)) = sum_c p(y(n,c) | x(n,:), theta(s))
        # P(n) = mean_s P(s,n)
        # Returns L(n) = log P(n)

        N = len(X)

        def logprob_fn(key):
            params_sample = self.sample_params(key, belief)
            # distribution  over batch size N
            pred_dist = self.predictive_distribution_given_params(params_sample, X)
            logprobs = pred_dist.log_prob(Y).reshape((N, -1))  # (N,1)
            return logprobs

        keys = random.split(key, nsamples_params)
        logprobs = vmap(logprob_fn)(keys)
        posterior_predictive_density = -jnp.log(nsamples_params)
        posterior_predictive_density += logsumexp(logprobs, axis=0)
        return posterior_predictive_density

    def joint_logprob_given_belief(self,
                                   key: chex.PRNGKey,
                                   belief: BeliefState,
                                   X: chex.Array,
                                   Y: chex.Array,
                                   nsamples_params: int):
        # X: T*N*D, Y: T*N*C, T = tau = size of joint event
        # P(t,s,n) = p(y(t,n,:) | x(t,n,:), theta(s))
        # Pjoint(s,n) = prod_t P(t,s,n)
        # Pavg(n) = mean_s Pjoint(s,n)
        # L(n) = log(Pavg(n))

        def logprob_fn(params_sample, X, Y):
            pred_dist = self.predictive_distribution_given_params(params_sample, X)
            logprobs = pred_dist.log_prob(Y)  # (N,1)
            return logprobs

        def logjoint_fn(key: chex.PRNGKey) -> float:
            params_sample = self.sample_params(key, belief)
            # distribution  over batch size N
            vlogprob = vmap(logprob_fn, in_axes=(None, 0, 0))
            logprobs = vlogprob(params_sample, X, Y)
            return jnp.sum(logprobs, axis=-1)

        keys = random.split(key, nsamples_params)
        logprobs_joint = vmap(logjoint_fn)(keys)

        posterior_predictive_density = -jnp.log(nsamples_params)
        posterior_predictive_density += logsumexp(logprobs_joint, axis=0)

        return posterior_predictive_density

    def posterior_predictive_mean(self,
                                    key: chex.PRNGKey,
                                    belief: BeliefState,
                                    x: chex.Array,
                                    nsamples_params: int,
                                    nsamples_output: int) -> chex.Array:

        samples = self.posterior_predictive_sample(key, belief, x, nsamples_params, nsamples_output)
        samples = jnp.transpose(samples, axes=[0, 2, 1, 3])
        _, _, batch_size, output_dim = samples.shape
        samples = samples.reshape((-1 , batch_size, output_dim))

        return jnp.mean(samples, axis=0)

    def posterior_predictive_mean_and_var(self,
                                    key: chex.PRNGKey,
                                    belief: BeliefState,
                                    x: chex.Array,
                                    nsamples_params: int,
                                    nsamples_output: int) -> chex.Array:
        samples = self.posterior_predictive_sample(key, belief, x, nsamples_params, nsamples_output)
        samples = jnp.transpose(samples, axes=[0, 2, 1, 3])
        _, _, batch_size, output_dim = samples.shape
        samples = samples.reshape((-1 , batch_size, output_dim))
        return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)