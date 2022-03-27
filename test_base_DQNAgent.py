
import rlcard
from rlcard.agents.dqn_agent import DQNAgent as DQNAgent

env = rlcard.make('blackjack')
env.set_agents([DQNAgent(num_actions=env.num_actions, mlp_layers=[2,2])])

print(env.num_actions) # 2
print(env.num_players) # 1
print(env.state_shape) # [[2]]
print(env.action_shape) # [None]

trajectories, payoffs = env.run()
