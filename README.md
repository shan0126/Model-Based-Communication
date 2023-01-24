# Model-Based-Communication

This is the code for "Model-based Sparse Communication in Multi-agent Reinforcement Learning," which is accepted in AAMAS 2023.

## Requirements
* OpenAI Gym 0.10
* PyTorch 1.5 (CPU)
* visdom

## Install Predator-Prey Environment

    cd pp_pixel\envs\ic3net-envs
    python setup.py develop


## Run the code

In predator-prey environment with grid information as agents’ observation (pp_grid).

    sh exp_pp_pixel.sh

In navigation environment with relative locations of landmarks and other agents as individual observations (cn_loc).

    sh exp_cn_loc.sh

In predator-prey environment with relative locations of preys and other predators as individual observation (pp_loc).

    sh exp_pp_loc.sh
    
## Reference
The training framework is adapted from MAGIC
