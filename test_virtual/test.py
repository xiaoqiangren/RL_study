# from stable_baselines import PPO2
# model = PPO2('MlpPolicy','overtaking-vo').learning(1000)

import gym
import highway_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make("overtaking-v0")
env.reset()
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
obs = env.reset()
for i in range(10000):
    action = env.action_type.actions_indexes["IDLE"]
   # action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
