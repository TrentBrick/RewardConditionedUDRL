import numpy as np 
from utils import ReplayBuffer, SortedBuffer


train_buffer = SortedBuffer(obs_dim=3, act_dim=1, size=100)

for e in range(30):
    amount_of_training_data = np.random.randint(10,12,1)[0]
    obs = np.random.random((amount_of_training_data,3))
    obs2 = obs 
    act = np.random.random((amount_of_training_data,1))
    rew = np.random.random((amount_of_training_data))
    terminal = np.random.random((amount_of_training_data))
    terminal_rew = np.repeat(np.random.randint(0,200), amount_of_training_data)
    rollout_length = np.repeat(np.random.randint(0,200), amount_of_training_data)
    horizon = np.repeat(np.random.randint(0,200), amount_of_training_data)
    train_data = [dict(terminal=terminal, rew=rew, 
        obs=obs, obs2=obs2, act=act, horizon=horizon, 
        cum_rew=terminal_rew, rollout_length=rollout_length )]
    train_buffer.add_rollouts(train_data)
    
    print('epoch', e, 'size of buffer', train_buffer.num_steps )
    print('==========')

from torch.utils.data import DataLoader, RandomSampler, BatchSampler

batch_size = 9
num_grad_steps = 13

bs = BatchSampler( RandomSampler(train_buffer, replacement=True, 
                    num_samples= num_grad_steps*batch_size  ), 
                    batch_size=batch_size, drop_last=False )
# testing if this can become a dataloader
dl = DataLoader(train_buffer, batch_sampler=bs) # batch_size=batch_size, shuffle=False)

for samp in dl:
    print(len(samp), samp['rew'], samp['obs'].shape )

print('===========')

print('getting desires')

print( train_buffer.get_desires(3)  )