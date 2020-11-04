# Reward Conditioned Policies / Upside Down Reinforcement Learning

This is an open source library that seeks to replicate the results from the papers: [Reward Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf) and [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877) neither of which shared their implementations.

This code base works for LunarLander in that the agent will learn to achieve a high score. However, performance is not as high as that documented in the original papers, this is especially the case for [Reward Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf). Even after correspondence with the authors (which was limited and slow) I have been unable to identify the bug or discrepancy in my code leading to such different performance.

There are a few other implementations of Upside Down Reinforcement Learning (UDRL) online already but these implementations either do not work or are very seed sensitive. This code base is not only more robust and performant across seeds but is also the first implementation of Reward Conditioned Policies. 

## Relevant Scripts:

* `train.py` - has almost all of the relevant configuration settings for the code. Also starts either ray tune (for hyperparam optimization) or a single model (for debugging). Able to switch between different model and learning types in a modular fashion
* `bash_train.sh` - uses GNU parallel to run multiple seeds of a model
* `lighting-trainer.py` - meat of the code. Uses pytorch lightning for training
* `control/agent.py` - runs rollouts of the environment and processes their rewards
* `envs/gym_params.py` - provides environment specific parameters
* `exp_dir/` - contains all experiments separated by: environment_name/experiment_name/seed/logged_versions
* `models/upsd_model.py` - contains the [Reward Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf) and [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877) upside down models.
* `models/advantage_model.py` - model to learn the advantage of actions as in the Levine paper

## Dependencies:
Tested with Python 3.7.5 (should work with Python 3.5 and higher).

Install Pytorch 1.7.0 (using CUDA or not depending on if you have a GPU)
<https://pytorch.org/get-started/locally/> 

If using Pip out of the box use: 
`pip3 install -r RewardConditionedUDRL/requirements.txt`

If using Conda then ensure pip3 is installed with conda and then run: 
`pip3 install -r RewardConditionedUDRL/requirements.txt`

## Running the code:

To run a single model of the LunarLander with the UDRL implementation call:

```
python trainer.py --implementation UDRL --gamename lunarlander \                                                  
--exp_name debug \
--num_workers 1 --no_reload --seed 25
```

Implementations are `UDRL`, `RCP-R` and `RCP-A` (Reward Conditioned Policies with Rewards and Advantages, respectively). For RCP the default is with exponential weighting rewards rather than advantages. 

Environments that are currently supported are `lunarlander` and `lunarlander-sparse`. Where the sparse version gives all of the rewards at the very end.

To run multiple seeds call `bash bash_train.sh` changing the trainer.py settings and experiment name as is desired.

To run Ray hyperparameter tuning, uncomment all of the `ray.tune()` functions for desired hyperparamters to search over and set `use_tune=True`.

## Evaluating Training

All training results along with important metrics are saved out to Tensorboard. To view them call: 

`tensorboard --logdir RewardConditionedUDRL/exp_dir/*ENVIRONMENT_NAME*/*EXPERIMENT_NAME*`

If you have run `python trainer.py` then the output will be in:
`tensorboard --logdir RewardConditionedUDRL/exp_dir/lunarlander/debug/seed_25/logger/`

To visualize the performance of a trained model, locate the model's checkpoint which will be under: `exp_dir/*ENVIRONMENT_NAME*/*EXPERIMENT_NAME*/*SEED*/epoch=*VALUE*.ckpt` and put this inside `load_name = join(game_dir, 'epoch=1940_v0.ckpt')` in trainer.py then call the code with with correct experiment name and `--eval 1` flag.

## Instructions for running on a Google Cloud VM:

#### Set up the VM: 
* Create a Google Cloud account
* Activate Free Credits
* Open up Compute Engine
* When it finishes setting up your compute engine go to "Quotas" and follow the instructions [here](https://stackoverflow.com/questions/45227064/how-to-request-gpu-quota-increase-in-google-cloud) to request 1 Global GPU.
* Wait for approval
* Create a new VM under Compute Engine. Choose "From Marketplace" on the left sidebar and search for "Deep Learning" choose the Google Deep Learning VM.
* Select a region (east-1d is the cheapest I have seen) and choose the T4 GPU (you can use others but will need to find the appropriate CUDA drivers that I list below yourself.)
* Select PyTorch (it has fast.ai too but we dont use this) and ensure it is CUDA 10.1
* For installation you can choose 1 CPU but at some point you will want to increase this to 16
* Select Ubuntu 16.04 as the OS
* Select you want the 3rd party driver software installed (as you will see later we install new drivers so this may be totally unnecessary but I did it and assume you have them installed in later steps)
* Add 150 GB of disk space
* Launch it.

The next two subheaders are if you want to be able to SSH into the server from your IDE (instructions provided for VS Code) (I recommend this!). But if you want to use the SSH button via Google Cloud thats fine too.

#### Connect Static IP 
In the top search bar look up "External IP" select it. Create a new static IP address. Attach it to your new VM.

(You may need to turn off and back on your VM for this to take effect.)

#### IDE SSH
For VSCode I use the installed plugin "Remote Explorer". 
My ssh keys are in ~/.ssh so I do `cat ~/.ssh/id_rsa.pub` and copy and paste this into the SSH section of Google Cloud (Search for SSH). 

Then with my server on I get its external IP address and in VSCode remote explorer call: 
`ssh -i ~/.ssh/id_rsa SSH_KEY_USERNAME@SERVER_IP_ADDRESS`
Before following the instructions.
One thing that first caught me up is that you need to give the ssh prefix not the the specific .pub file!

## TODOs

Enable multicore training and Gym rollouts (would probably be best to use Ray RL package for this.)



## Acknowledgements

Thanks to the open source implementation of Upside Down Reinforcement Learning: <https://github.com/jscriptcoder/Upside-Down-Reinforcement-Learning> which provided an initial test base. Also to [Reward Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf) and [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877) for initial research and results (I just wish both of these papers shared their code...).

## Authors

* **Trenton Bricken** - [trentbrick](https://github.com/trentbrick)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details