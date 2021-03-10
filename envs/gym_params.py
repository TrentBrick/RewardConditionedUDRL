import torch


def get_env_params(gamename):

    if gamename == "carracing":
        env_params = {
            'env_name': 'CarRacing-v0',
            'desired_horizon': 30,
            'sparse':False,
            'num_action_repeats': 5,
            'avg_episode_length':1000, #NB not sure if this is right!
            'time_limit':None, # max time limit for the rollouts generated
            'NUM_IMG_CHANNELS': 3,
            'ACTION_SIZE': 3,
            'STORED_ACTION_SIZE': 1,
            'init_cem_params': ( torch.Tensor([0.,0.7,0.]), 
                        torch.Tensor([0.5,0.7,0.3]) ),
            'LATENT_SIZE': 32, 
            'LATENT_RECURRENT_SIZE': 512,
            'EMBEDDING_SIZE': 256,
            'NODE_SIZE': 256,
            'IMAGE_RESIZE_DIM': 64,
            'IMAGE_DEFAULT_SIZE': 96,
            # top, bottom, left, right
            # can set to equal None if dont want any trimming. 
            'trim_shape': (0,84,0,96), 
            'give_raw_pixels':False, # for environments that are otherwise state based. 
            'use_vae':True,
            'reward_prior_mu': 4.0, 
            'reward_prior_sigma':0.1
        }

    elif gamename == "pendulum":
        env_params = {
            'env_name': 'Pendulum-v0',
            'continuous_actions':True,
            'sparse':False,
            'desired_reward':10,
            'desired_horizon': 30,
            'num_action_repeats': 1,
            'avg_episode_length':200, # NB not sure if this is right!
            'time_limit':None, # max time limit for the rollouts generated
            'NUM_IMG_CHANNELS': 3,
            'ACTION_SIZE': 1,
            'STORED_ACTION_SIZE': 1,
            'STORED_STATE_SIZE': 3,
            'init_cem_params': ( torch.Tensor([0.]), 
                        torch.Tensor([2.]) ),
            'LATENT_SIZE': 3, 
            'LATENT_RECURRENT_SIZE': 256,
            'EMBEDDING_SIZE': 3,
            'NODE_SIZE': 128,
            'IMAGE_RESIZE_DIM': 64,
            'IMAGE_DEFAULT_SIZE': 96,
            'desires_size' : 1, # just reward for now. 
            # top, bottom, left, right
            # can set to equal None if dont want any trimming. 
            'trim_shape': None,
            'give_raw_pixels':False,
            'use_vae':False, 
            'reward_prior_mu': 0.0, 
            'reward_prior_sigma':0.2,
            'action_noise' :0.3
        }

    elif gamename == "lunarlander" or gamename == "lunarlander-sparse":
        env_params = {
            'env_name': 'LunarLander-v2',
            'sparse':False,
            'continuous_actions':False,
            'desired_reward':200,
            'num_action_repeats': 1,
            'avg_episode_length':200,
            'time_limit':None, # max time limit for the rollouts generated
            'over_max_time_limit_penalty':None,
            'ACTION_SIZE': 4, # number possible actions
            'STORED_ACTION_SIZE': 1,
            'STORED_STATE_SIZE': 8,
            'max_reward':320,
            # top, bottom, left, right
            # can set to equal None if dont want any trimming. 
            'trim_shape': None,
            'give_raw_pixels':False,
            'use_vae':False, 
            'action_noise' : 1.0#0.05 # lower the value the more uniform it is
        }
        if gamename == "lunarlander-sparse":
            env_params['sparse'] = True

    elif gamename == "cartpole":
        env_params = {
            'desired_horizon': 30,
            'num_action_repeats': 3,
            'avg_episode_length':200, # NB not sure if this is right!
            'NUM_IMG_CHANNELS': 3,
            'sparse':False,
            'ACTION_SIZE': 3,
            'STORED_ACTION_SIZE': 1,
            'init_cem_params': ( torch.Tensor([0.,0.,0.]), 
                torch.Tensor([1.,1.,1.]) ),
            'LATENT_SIZE': 32, 
            'LATENT_RECURRENT_SIZE': 512,
            'IMAGE_RESIZE_DIM': 64,
            'IMAGE_DEFAULT_SIZE': 96,
        }

    else: 
        raise ValueError("Don't know what this gamename (environment) is!")


    return env_params
