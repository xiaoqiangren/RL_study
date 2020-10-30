# from stable_baselines import PPO2
# import highway_env
# model = PPO2('MlpPolicy','overtaking-v0').learn(100)

import gym
import highway_env
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy
# from stable_baselines.deepq import DQNPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG
from stable_baselines import DQN

env = gym.make("overtaking-v0")
env.reset()
model = PPO2(MlpPolicy, env, verbose=1)
# model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
obs = env.reset()
for i in range(100):
    action = env.action_type.actions_indexes["IDLE"]
    # action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
