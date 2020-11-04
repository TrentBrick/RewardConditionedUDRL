# pylint: disable=no-member
""" 
Training of the UpsideDown RL model.
"""
import argparse
import time
from functools import partial
from os.path import join, exists
from os import mkdir, unlink
import torch
import torch.nn.functional as F 
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
from tqdm import tqdm
from envs import get_env_params
import sys
from lightning_trainer import LightningTemplate
from multiprocessing import cpu_count
from collections import OrderedDict
from utils import ReplayBuffer, \
    RingBuffer, SortedBuffer
import time 
import random 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

class TuneReportCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics["policy_loss"],
            mean_reward=pl_module.mean_reward_rollouts,
            mean_reward_20_epochs = sum(pl_module.mean_reward_over_20_epochs[-20:])/20,
            epoch=trainer.current_epoch)

def main(args):

    # get environment parameters: 
    env_params = get_env_params(args.gamename)

    assert args.num_workers <= cpu_count(), "Providing too many workers!"

    # Constants
    epochs = 2000  

    if args.implementation == "UDRL":

        imp = dict(
            use_RCP_desire_sampling=False, 
            use_RCP_buffer = False, 
            use_RCP_model = False, 
            use_exp_weight_losses = False,

            desire_discounted_rew_to_go = False, #tune.grid_search( [True, False]),
            desire_cum_rew = True, #tune.grid_search( [True, False]), # mutually exclusive to discounted rewards to go. 
            desire_horizon = True, #tune.grid_search( [True, False]),
            discount_factor = 1.0, #ie. no discount
            
            desire_advantage = False, #tune.grid_search( [True, False]), 
            use_lambda_td = False,  
            td_lambda = 0.95,
        )

    elif 'RCP' in args.implementation:
        imp = dict(
            use_RCP_desire_sampling=True, 
            use_RCP_buffer = True, 
            use_RCP_model = True, 
            
            desire_cum_rew = False, #tune.grid_search( [True, False]), # mutually exclusive to discounted rewards to go. 
            desire_horizon = False, #tune.grid_search( [True, False]),
            discount_factor = 0.99,

            use_exp_weight_losses = args.no_expo_weighting,
            beta_reward_weighting = 1.0, # value in the RCP weighting equation. 
            max_loss_weighting = 20 # max exponential weighting to a given loss
        )
        
        if args.implementation == "RCP-R":
            imp.update( dict(
                desire_discounted_rew_to_go = True, #tune.grid_search( [True, False]),
                desire_advantage = False, #tune.grid_search( [True, False]), 
                use_lambda_td = False,  # was experimenting with not using TD lambda for the value function. 
                td_lambda = 0.95,
            ))
            
        elif args.implementation == "RCP-A":
            imp.update(dict(
                desire_discounted_rew_to_go = False, #tune.grid_search( [True, False]),
                desire_advantage = True, #tune.grid_search( [True, False]), 
                use_lambda_td = True,  
                td_lambda = 0.95,
                clamp_adv_to_max = False, # was experimenting with ways to make the advantage training more stable. 
                max_adv_clamp_val = 50
            ))

    config = dict(
        random_action_epochs = 1, # needs to be at least 1 to initalize the data buffer
        val_func_update_iterval = 5, # for every ... updates of the policy, update the value function
        grad_clip_val = 100, 
        eval_every = 10, # evaluate (run with argmax on actions rather than sampling/noise (discrete/continous action space respectively)) every ... epochs.
        eval_episodes=10, # how many environment rollouts to run for each evaluation
        training_rollouts_per_worker = 20, #tune.grid_search( [10, 20, 30, 40]),
        num_rand_action_rollouts = 50, # increased this by quite a bit!
        antithetic = False, # TODO: implement this!
        num_val_batches = 2, # number of mini-batches of data to draw for the validation samples
        desire_mu_minus_std = False # experimenting with different distributions to sample the next desires from during rollouts. 
    )

    # loading implmentation details into the config dictionary. 
    config.update(imp)

    # used throughout to ensure the order of desires is consistent.
    # advantage needs to go last because it is computed on the fly 
    # in the training loop.  
    config['desires_official_order'] = ['desire_discounted_rew_to_go',
    'desire_cum_rew', 'desire_horizon', 'desire_advantage']

    args_dict = vars(args)

    # used for Ray tune where permutations of desires are made. 
    # If none of the desires are on will kill job with a clear error message. 
    num_on = 0
    for k in config['desires_official_order']:
        num_on += config[k]
    if num_on == 0: 
        raise Exception('No desires turned on! Killing this job!')
    
    if config['desire_cum_rew'] and config['desire_discounted_rew_to_go']:
        raise Exception("Cant have both of these live at the same time.")

    if config['use_RCP_model']:
        model_params = dict(
            lr= 0.001, #tune.grid_search(np.logspace(-4, -2, num = 101)),
            hidden_sizes = [64,64],#[128,128,64],
            desire_scalings =False,
            num_grad_steps = 1000
        )
    else: 
        model_params = dict(
            lr= 0.001, #tune.grid_search(np.logspace(-4, -2, num = 101)),
            hidden_sizes = [32,64], #[32,64,64,64],
            desire_scalings =True,
            horizon_scale = 0.01, #tune.grid_search( [0.01, 0.015, 0.02, 0.025, 0.03]), #(0.02, 0.01), # reward then horizon
            reward_scale = 0.01, #tune.grid_search( [0.01, 0.015, 0.02, 0.025, 0.03]),
            state_scale = 1.0,
            num_grad_steps = 100
            #tune.grid_search([[32], [32, 32], [32, 64], [32, 64, 64], [32, 64, 64, 64],
            #[64], [64, 64], [64, 128], [64, 128, 128], [64, 128, 128, 128]])
        )

    if config['use_RCP_buffer']:
        config['batch_size'] = 256
        config['max_buffer_size'] = 100000
    else: 
        config['batch_size'] = 768 #tune.grid_search([512, 768, 1024, 1536, 2048]),
        config['max_buffer_size'] = 250 #tune.grid_search([300, 400, 500, 600, 700]),
        config['last_few'] = 25 #tune.grid_search([25, 75]),

    config.update(args_dict)
    config.update(env_params)
    config.update(model_params)

    # TODO: do I need to scale different parts of this differently? 
    if config['desire_scalings']:
        config['state_scale'] = np.repeat(config['state_scale'], env_params['STORED_STATE_SIZE']).tolist()

    # Load in the Model, Loggers, etc:
    if args.use_tune:
        game_dir = ''
        run_name = str(np.random.randint(0,1000,1)[0])
        logger=False 
        every_checkpoint_callback = False 
        callback_list = [TuneReportCallback()]
        config['seed'] = tune.grid_search([25,26,27,28,29])
    else: 
        # Init save filenames 
        base_game_dir = join(args.logdir, args.gamename)
        exp_dir = join(base_game_dir, args.exp_name)
        game_dir = join(exp_dir, 'seed_'+str(args.seed))
        filenames_dict = { bc:join(game_dir, 'model_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
        for dirr in [base_game_dir, exp_dir, game_dir]:
            if not exists(dirr):
                mkdir(dirr)

        logger = TensorBoardLogger(game_dir, "logger")
        # have the checkpoint overwrite itself. 
        every_checkpoint_callback = False 
        # uncomment if want to save checkpoints of model being trained. 
        '''every_checkpoint_callback = ModelCheckpoint(
            filepath=game_dir,
            save_top_k=1,
            verbose=False ,
            monitor='policy_loss',
            mode='min',
            prefix=''
        )'''
        callback_list = []
    '''best_checkpoint_callback = ModelCheckpoint(
        filepath=filenames_dict['best'],
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )'''

    def run_lightning(config):

        if config['use_RCP_buffer']:
            train_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
                act_dim=env_params['STORED_ACTION_SIZE'], 
                size=config['max_buffer_size'], use_td_lambda_buf=config['desire_advantage'])
            test_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
                act_dim=env_params['STORED_ACTION_SIZE'], 
                size=config['batch_size']*10, use_td_lambda_buf=config['desire_advantage'])
        else:
            config['max_buffer_size'] *= env_params['avg_episode_length']
            train_buffer = SortedBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
                act_dim=env_params['STORED_ACTION_SIZE'], 
                size=config['max_buffer_size'], 
                use_td_lambda_buf=config['desire_advantage'] )
            test_buffer = SortedBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
                act_dim=env_params['STORED_ACTION_SIZE'], 
                size=config['batch_size']*10,
                use_td_lambda_buf=config['desire_advantage'])

        model = LightningTemplate(game_dir, config, train_buffer, test_buffer)

        if not args.no_reload:
            # load in trained model: 
            # TODO: enable plugging in experiment name from the command line rather than in here!
            load_name = join(game_dir, 'epoch=1940_v0.ckpt')
            print('loading in from:', load_name)
            state_dict = torch.load(load_name)['state_dict']
            state_dict = {k[6:]:v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
            print("Loaded in Model state!")

        if args.eval_agent:
            model.eval_agent()

        else: 
            trainer = Trainer(deterministic=True, logger=logger,
                default_root_dir=game_dir, max_epochs=epochs, profiler=False,
                checkpoint_callback = every_checkpoint_callback,
                callbacks=callback_list, 
                gradient_clip_val=config['grad_clip_val'], 
                progress_bar_refresh_rate=0
            )
            trainer.fit(model)

    # Ray Tune settings: 
    scheduler = ASHAScheduler(
        time_attr='epoch',
        # trailing 20 epochs of mean reward obtained by rollouts (trailing to reduce noise)
        metric="mean_reward_20_epochs",
        mode="max",
        # max amount of time. 
        max_t=epochs,
        # time to run before killing low performing jobs
        grace_period=1000,
        reduction_factor=4)

    reporter = CLIReporter(
        metric_columns=["loss", "mean_reward_20_epochs", "epoch"],
        )

    num_samples = 1
    if args.use_tune:
        tune.run(
            run_lightning,
            name=run_name,
            resources_per_trial={"cpu": 1},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
            fail_fast=False )

    else: 
        run_lightning(config)
        
if __name__ =='__main__':
    parser = argparse.ArgumentParser("Training Script")
    parser.add_argument('--implementation', type=str, default='UDRL',
                        help="Whether or not want the RCP-R, RCP-A or UDRL Implementation.")
    parser.add_argument('--no_expo_weighting', action='store_false',
                        help="If using RCP-A or RCP-R, whether or not to use the exponential weighting in the loss. On by default. ")
    
    
    parser.add_argument('--gamename', type=str, default='lunarlander',
                        help="What Gym environment to train in.")
    parser.add_argument('--exp_name', type=str, default='debug',
                        help="Name of the experiment.")                
    parser.add_argument('--logdir', type=str, default='exp_dir',
                        help="Where things are logged and models are loaded from.")
    
    parser.add_argument('--no_reload', action='store_true', default=True,
                        help="Won't load in models for MODEL from the joint file. \
                        NB. This will create new models with random inits and will overwrite \
                        the best and checkpoints!")
    parser.add_argument('--giving_pretrained', action='store_true',
                        help="If pretrained models are being provided, avoids loading in an optimizer \
                        or previous lowest loss score.")
    
    parser.add_argument('--num_workers', type=int, help='Maximum number of workers.',
                        default=1)
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    parser.add_argument('--seed', type=int, default=25,
                        help="Starter seed for reproducible results")
    parser.add_argument('--eval_agent', type=bool, default=False,
                        help="Able to eval the agent!")

    parser.add_argument('--use_tune', action='store_true',
                        help="Whether or not to use Ray Tune")

    parser.add_argument('--print_statements', type=int, default=0,
                        help="Able to eval the agent!")

    args = parser.parse_args()
    main(args)