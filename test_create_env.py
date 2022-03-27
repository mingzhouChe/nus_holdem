# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:43:08 2022

@author: mingz
"""

import rlcard

env = rlcard.make('limit-holdem', config={
    'seed': 123
    ,'allow_step_back': True
    })

env = rlcard.make('no-limit-holdem', config={
    'seed': 123
    ,'allow_step_back': True
    })

from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent

device = get_device()
agent = DQNAgent(num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 mlp_layers=[64,64],
                 device=device)

agents = [agent]

agents = []
for _ in range(0, env.num_players):
    agents.append(RandomAgent(num_actions=env.num_actions))
env.set_agents(agents)

all_trajectories = []
for i in range(5000):
    trajectories, payoffs = env.run(is_training=True)
    # Reorganaize the data to be state, action, reward, next_state, done
    trajectories = reorganize(trajectories, payoffs)
    all_trajectories.extend(trajectories)


# Feed transitions into agent memory, and train the agent
# Here, we assume that DQN always plays the first position
# and the other players play randomly (if any)
for ts in trajectories[0]:
    agent.feed(ts)

tournament(env, 20)
tournament(env, 2000)
tournament(env, 5000)

len(trajectories)

trajectories[0][0]
trajectories[1][0]

trajectories[0][1]
trajectories[1][1]

trajectories[0][2]
trajectories[1][2]

payoffs



trajectories_2 = reorganize(trajectories, payoffs)

len(trajectories_2)
trajectories_2[0][0]
trajectories_2[1][0]
trajectories_2[1][0]
trajectories_2[1][1]


# one example
state, player_id = env.reset()
action = env.agents[player_id].step(state)
# Environment steps
next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
# another player take action
action = env.agents[player_id].step(state)
next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)

action = 1#env.agents[player_id].step(state)
next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)















