
import rlcard

env = rlcard.make('limit-holdem', config={
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
for _ in range(0, env.num_players):
    agents.append(RandomAgent(num_actions=env.num_actions))
env.set_agents(agents)

all_trajectories = []
for i in range(5000):
    trajectories, payoffs = env.run(is_training=True)
    # Reorganaize the data to be state, action, reward, next_state, done
    trajectories = reorganize(trajectories, payoffs)
    all_trajectories.extend(trajectories)

tournament(env, 20)
tournament(env, 2000)
tournament(env, 5000)
tournament(env, 10000)
