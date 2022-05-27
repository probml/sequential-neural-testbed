from jax import random
from enn_experiments.agents.sgld_agent import SGLDConfig, make_sgmcmcjax_agent
from enn_experiments.agents.base import PriorKnowledge
from enn_experiments.demos.mcmc_agent_demo import make_dataset, make_plot


def main():
    n = 100
    minval, maxval = 0.0, 0.5
    key = random.PRNGKey(0)
    dataset = make_dataset(key,
                        n,
                        minval,
                        maxval)
    input_dim, tau = 1, 1
    noise_std = 1.
    output_sizes = [16, 16, 1]
    prior = PriorKnowledge(input_dim,
                          n,
                          tau,
                          noise_std=noise_std,
                          output_sizes=output_sizes)
    dt = 1e-6
    
    config = SGLDConfig(dt, n)
    sgld_agent = make_sgmcmcjax_agent(config, prior)
    sampler = sgld_agent(dataset)
    
    dataset_kwargs = { "n": n,
                    "minval" : minval,
                    "maxval" : maxval,
                    "key" : random.PRNGKey(0)
                 }
    p = make_plot(sampler,
                dataset_kwargs=dataset_kwargs)
    _ = p.draw()

if __name__ == '__main__':
    main()