# pylint: disable=no-member
import math
import random 
import time
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
import pickle
import gym 
import gym.envs.box2d
from torch.distributions import Normal, Categorical 
#from utils import sample_mdrnn_latent
from models import UpsdModel, UpsdBehavior
from envs import get_env_params
import random 
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class WeightedNormal:
    def __init__(self, mu, sigma, beta=1):
        self.normal = Normal(mu, sigma)
        self.beta = beta

    def sample(self, nsamps):
        s = self.normal.sample([nsamps])
        s = s*torch.exp(s/self.beta)
        #print("desired reward sampled", s)
        return s

import scipy.signal
def discount_cumsum(x, discount):
    # Taken from OpenAI spinning up. 
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Agent:
    def __init__(self, gamename,
        model = None, 
        hparams=None, 
        take_rand_actions = False,
        desire_dict = None, 
        model_version = 'checkpoint',
        return_plan_images=False,
        advantage_model=None):
        """ Runs and collects rollouts """

        self.gamename = gamename
        self.hparams = hparams
        self.env_params = get_env_params(gamename)
        self.action_noise = self.env_params['action_noise']
        self.take_rand_actions = take_rand_actions
        self.discount_factor = self.hparams['discount_factor']
        self.advantage_model = advantage_model

        self.desire_dict = desire_dict
        self.td_lambda = self.hparams['td_lambda']

        # top, bottom, left, right
        self.obs_trim = self.env_params['trim_shape']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = self.env_params['time_limit']
        self.num_action_repeats = self.env_params['num_action_repeats']

        if model:
            self.model = model 
            self.model.eval()

        # can be set to true inside Simulate. 
        self.return_events = False

    def make_env(self, seed, render_mode=False, full_episode=False):
        """ Called every time a new rollout occurs. Creates a new environment and 
        sets a new random seed for it."""
        self.render_mode = render_mode
        self.env = gym.make(self.env_params['env_name'])
        self.env.reset()
        #if not seed: 
        #seed = np.random.randint(0,1e9,1)[0]

        self.env.seed(int(seed))
        self.env.action_space.np_random.seed(int(seed))
        if render_mode: 
            self.env.render(mode='rgb_array')

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action

    def constrain_actions(self, actions):
        """ Ensures actions sampled from the gaussians are within the game bounds."""
        for ind, (l, h) in enumerate(zip(self.env.action_space.low, self.env.action_space.high)):
            actions[:,ind] = torch.clamp(actions[:,ind], min=l, max=h)
        return actions

    def rollout(self, render=False, display_monitor=None,
            greedy=False, recorder = None):
        """ Executes a rollout and returns cumulative reward along with
        the time point the rollout stopped and 
        optionally, all observations/actions/rewards/terminal outputted 
        by the environment.

        :args:
            - rand_env_seed: int. Used to help guarentee this rollout is random.

        :returns: (cumulative, t, [rollout_dict])
            - cumulative: float. cumulative reward
            - time: int. timestep of termination
            - rollout_dict: OPTIONAL. Dictionary with keys: 'obs', 'rewards',
                'actions', 'terminal'. Each is a PyTorch Tensor with the first
                dimension corresponding to time. 
        """

        obs = self.env.reset()
        if self.env_params['give_raw_pixels']:
            obs = self.env.render(mode='rgb_array')
            #self.env.viewer.window.dispatch_events()
        
        # initialize all of the desires. 
        if not self.take_rand_actions:
            current_desires_dict = dict()
            if self.hparams['desire_discounted_rew_to_go'] or self.hparams['desire_cum_rew']:
                if self.hparams['use_RCP_desire_sampling']:
                    init_rew = Normal(self.desire_dict['reward_dist'][0], 
                                        self.desire_dict['reward_dist'][1]).sample([1])
                else: 
                    init_rew = torch.Tensor([min(np.random.uniform(self.desire_dict['reward_dist'][0], self.desire_dict['reward_dist'][0]+self.desire_dict['reward_dist'][1]), self.env_params['max_reward']  )])
                    
            # cumulative and reward to go start the same/sampled the same. then RTG is 
            # annealed. 
            if self.hparams['desire_discounted_rew_to_go']:
                current_desires_dict['discounted_rew_to_go'] = init_rew
            
            if self.hparams['desire_cum_rew']:
                current_desires_dict['cum_rew'] = init_rew

            if self.hparams['desire_horizon']:
                current_desires_dict['horizon'] = torch.Tensor([self.desire_dict['horizon']])
                
            if self.hparams['desire_advantage']:
                current_desires_dict['advantage'] = Normal(self.desire_dict['advantage_dist'][0], 
                                        self.desire_dict['advantage_dist'][1]).sample([1])

        if self.gamename == 'carracing':
            sim_rewards_queue = []
        cumulative = 0
        time = 0
        hit_done = False

        if self.return_events:
            rollout_dict = {k:[] for k in ['obs', 'obs2', 'rew', 'act', 'terminal']}
        
        while not hit_done:

            if recorder is not None:
                recorder.capture_frame()

            # NOTE: maybe make this unique to carracing here? 
            if self.obs_trim is not None:
                # trims the control panel at the base for the carracing environment. 
                obs = obs[self.obs_trim[0]:self.obs_trim[1], 
                            self.obs_trim[2]:self.obs_trim[3], :]

            if render: 
                if display_monitor:
                    display_monitor.set_data(obs)
                self.env.render()

            obs = torch.Tensor(obs).unsqueeze(0)#.to(self.device)

            if self.take_rand_actions:
                action = self.env.action_space.sample()
            else: 

                desires = []
                for key in self.hparams['desires_order']:
                    desires.append( current_desires_dict[key.split('desire_')[-1]].unsqueeze(1) )
                    
                action = self.model(obs, desires )

                # need to constrain the action! 
                if self.env_params['continuous_actions']:
                    if not greedy: 
                        action = self._add_action_noise(action, self.action_noise)
                    action = self.constrain_actions(action)
                    action = action[0].detach().numpy()
                else: 
                    #sample action
                    # to do add temperature noise. 
                    #print('action is:', action)
                    
                    if greedy: 
                        action = torch.argmax(action).squeeze().detach().numpy()
                    else: 
                        action = torch.softmax(action*self.action_noise, dim=-1)
                        action = Categorical(probs=action).sample([1])
                        action = action.squeeze().detach().numpy()
            
            # using action repeats
            action_rep_rewards = 0
            for _ in range(self.num_action_repeats):
                next_obs, reward, done, _ = self.env.step(action)
                action_rep_rewards += reward
                # ensures the done indicator is not missed during the action repeats.
                if done: hit_done = True

            # reward is all of the rewards collected during the action repeats. 
            reward = action_rep_rewards

            if self.env_params['give_raw_pixels']:
                next_obs = self.env.render(mode='rgb_array')
                #self.env.viewer.window.dispatch_events()

            if self.time_limit is not None: 
                if not hit_done and time>=self.time_limit:
                    # add in any penalty for hitting the time limit and still not being done. 
                    reward += self.env_params['over_max_time_limit_penalty']

                if time >= self.time_limit:
                    hit_done=True
            
            time += 1
                
            # if the last n steps (real steps independent of action repeats)
            # have all given -0.1 reward then cut the rollout early. 
            if self.gamename == 'carracing':
                if len(sim_rewards_queue) < 50:
                    sim_rewards_queue.append(reward)
                else: 
                    sim_rewards_queue.pop(0)
                    sim_rewards_queue.append(reward)
                    #print('lenght of sim rewards',  len(sim_rewards_queue),round(sum(sim_rewards_queue),3))
                    if round(sum(sim_rewards_queue), 3) == -5.0:
                        hit_done=True
            # done checking for hit_done. 

            cumulative += reward
            if self.env_params['sparse']:
                # store in cumulative first. Dont need to worry about
                # storing in cumulative later as reward set to 0!
                reward = cumulative if hit_done else 0.0

            # update reward desires! 
            if not self.take_rand_actions:

                if self.hparams['desire_discounted_rew_to_go']:
                    if self.hparams['use_RCP_desire_sampling']:
                        pass 
                    else: 
                        # lower the desired reward.  
                        current_desires_dict['discounted_rew_to_go'] = torch.Tensor( [min(current_desires_dict['discounted_rew_to_go']-reward, self.env_params['max_reward'])])

                if self.hparams['desire_cum_rew']:
                    # don't do any updating. 
                    pass

                if self.hparams['desire_horizon']:
                    current_desires_dict['horizon'] = torch.Tensor ( [max( current_desires_dict['horizon']-1, 1)])
                
                if self.hparams['desire_advantage']:
                    current_desires_dict['advantage'] = Normal(self.desire_dict['advantage_dist'][0], 
                                            self.desire_dict['advantage_dist'][1]).sample([1])

            # save out things.
            # doesnt save out the time so dont need to worry about it here. 
            if self.return_events:
                for key, var in zip(['obs', 'obs2', 'rew', 'act', 'terminal'], 
                                        [obs, next_obs, reward, action, hit_done ]):
                    if key=='obs':
                        var = var.squeeze().detach().numpy()
                    rollout_dict[key].append(var)
                
            # This is crucial. 
            obs = next_obs

        if recorder is not None:
            recorder.close()
            #del recorder

        if render: 
            print('the last state for agent is:', rollout_dict['obs'][-1].round(3)  )

        if self.return_events:
            list_of_keys = list(rollout_dict.keys())
            for k in list_of_keys: # list of tensors arrays.
                # rewards to go here for RCP
                if k =='rew':
                    if self.advantage_model:
                        rollout_dict['raw_rew'] = np.asarray(rollout_dict[k]) # need for TD lambda

                    # discounted rewards to go
                    rollout_dict['discounted_rew_to_go'] = discount_cumsum(np.asarray(rollout_dict[k]), self.discount_factor)
                    to_desire = rollout_dict['discounted_rew_to_go'][0]
                else: 
                    rollout_dict[k] = np.asarray(rollout_dict[k])
                    
            # repeat the cum reward up to length times. 
            # setting cum_rew to be the first reward to go. This is equivalent to the cum reward 
            # but accounts too for any discounting factor.
            rollout_dict['cum_rew'] = np.repeat(cumulative, time)
            rollout_dict['rollout_length'] = np.repeat(time, time)
            rollout_dict['horizon'] = time - np.arange(0, time) 
            rollout_dict['final_obs'] = np.repeat(np.expand_dims(rollout_dict['obs'][-1],0), time, axis=0)
            # discounted cumulative!
            return cumulative, to_desire, time, rollout_dict # passed back to simulate. 
        else: 
            return cumulative, time # ending time and cum reward
                
    def simulate(self, seed, return_events=False, num_episodes=16, 
        render_mode=False, antithetic=False, greedy=False,
        record_path = None):
        """ Runs lots of rollouts with different random seeds. Optionally,
        can keep track of all outputs from the environment in each rollout 
        (adding each to a list which contains dictionaries for the rollout). 
        And it can compute the FEEF at the end of each rollout. 
        
        :returns: (cum_reward_list, terminal_time_list, [data_dict_list], [feef_losses_list])
            - cum_reward_list: list. cumulative rewards of each rollout. 
            - terminal_time_list: list. timestep the rollout ended at. 
            - data_dict_list: list. OPTIONAL. Dictionaries from each rollout.
                Has keys: 'obs', 'rewards',
                'actions', 'terminal'. Each is a PyTorch Tensor with the first
                dimension corresponding to time.  
            - feef_losses_list: list. OPTIONAL. Free Energy of Expected Future value
            from the whole rollout. 
        """

        self.return_events = return_events

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.make_env(seed=seed)

        cum_reward_list = []
        terminal_time_list = []
        to_desire_list = []
        if self.return_events:
            data_dict_list = []

        with torch.no_grad():
            for i in range(num_episodes):
                if record_path and i < self.hparams['record_n_rollouts_per_epoch']:
                    # record the first n rollouts of this epoch. 
                    recorder = VideoRecorder(self.env, path=record_path+'/rollout_'+str(i)+'.mp4')
                else: 
                    recorder=None

                # for every second rollout. reset the rand seed if using antithetic. 
                if antithetic and i%2==1:
                    # uses the previous rand_seed
                    self.env.seed(int(rand_env_seed))
                    rand_env_seed = np.random.randint(0,1e9,1)[0]
                else: 
                    rand_env_seed = np.random.randint(0,1e9,1)[0]
                    self.env.seed(int(rand_env_seed))

                if self.return_events: 
                    rew, to_desire, time, data_dict = self.rollout(render=render_mode, greedy=greedy, recorder=recorder)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                    data_dict_list.append(data_dict)
                    to_desire_list.append(to_desire)
                else: 
                    rew, time = self.rollout(render=render_mode, greedy=greedy,recorder=recorder)
                if render_mode: 
                    print('Cumulative Reward is:', rew, 'Termination time is:', time)
                    #print('Last Desired reward is:',curr_des "Last Desired Horizon is:", )
                cum_reward_list.append(rew)
                terminal_time_list.append(time)

        self.env.close()

        if render_mode:
            print('Mean reward over', num_episodes, 'episodes is:', np.mean(cum_reward_list))

        if self.return_events: 
            return cum_reward_list, to_desire_list, terminal_time_list, data_dict_list 
        else: 
            return cum_reward_list, terminal_time_list