import gym
import numpy as np
from gym.envs.registration import register

# Refer https://github.com/openai/gym/issues/565
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=2000,
    reward_threshold=0.78, # optimum = .8196
)

#env = gym.make("FrozenLakeNotSlippery-v0")
env = gym.make("FrozenLakeNotSlippery-v0")
env.seed(0)
np.random.seed(56776)

# Test how does the game work.
print("-------------Test game--------------")
ql_table = np.zeros([env.observation_space.n, env.action_space.n])
print(ql_table)
env.render()
env.reset()

hardcore_steps = [1, 1, 2, 2, 1, 2]
for step in hardcore_steps:
    env.step(step)
    env.render()
    
    
# Let machine learng the step.
print("-------------Let machine learng the steps--------------")
env.reset()
env.render()

ql_table = np.zeros([env.observation_space.n, env.action_space.n]) + np.random.randn(16, 4)
print(ql_table)

"""
Hyper parameters:
"""
n_round = 5000
n_steps = 2000
lr = 0.3
discount = 0.8

for round in range(n_round):
    state = env.reset()
    
    for step in range(n_steps):
        action = np.argmax(ql_table[state, :] + np.random.randn(1, 4))
        new_state, reward, done, _ = env.step(action)
        ql_table[state, action] = (1 - lr) * ql_table[state, action] + \
                                  lr * (reward + discount * np.max(ql_table[new_state, :]))
        
        state = new_state
        
        if done is True:
            break

print(np.argmax(ql_table, axis=1))
print(np.around(ql_table, 6))

env.reset()
for step in np.argmax(ql_table, axis=1):
    state_new, reward, done, _ = env.step(step)
    env.render()