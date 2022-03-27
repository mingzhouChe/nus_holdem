# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:36:41 2022

@author: Ark
"""


    


import rlcard

env = rlcard.make('limit-holdem', config={
    'seed': 123
    ,'allow_step_back': True
    ,'game_num_players': 2
    })

import OdssAgentV1
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1

device = get_device()

#DQN 
agent1 = DQNAgent(num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 mlp_layers=[64,64],
                 device=device)

#Odss
agent1 = OdssAgentV1(num_actions=env.num_actions)

#NFSP
agent1 = NFSPAgent(num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 hidden_layers_sizes = [4],
                 q_mlp_layers=[64,64],
                 device=device)


agents = []
for _ in range(0, env.num_players-1):
    agents.append(agent1)#RandomAgent(num_actions=env.num_actions))

for _ in range(0, 1):
    agents.append(RandomAgent(num_actions=env.num_actions))    
    
env.set_agents(agents)


# test without training
all_trajectories = []
for i in range(5000):
    trajectories, payoffs = env.run(is_training=False)
    # Reorganaize the data to be state, action, reward, next_state, done
    trajectories = reorganize(trajectories, payoffs)
    all_trajectories.extend(trajectories)
  
    
# test with training
all_trajectories = []
for i in range(5000):
    trajectories, payoffs = env.run(is_training=True)
    # Reorganaize the data to be state, action, reward, next_state, done
    trajectories = reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
        agent1.feed(ts)
    all_trajectories.extend(trajectories)

print('\n')
for i in range(0,5):
    print(tournament(env, 1000))










