import os

import gym
import highway_env
import pybullet_envs

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make("overtaking-v0")])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)


model = PPO2('MlpPolicy', env)
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/home/cxc/下载/实验结果"
model.save(log_dir + "ppo_overtaking")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# To demonstrate loading
del model, env

# Load the agent
model = PPO2.load(log_dir + "ppo_overtaking")



# Load the saved statistics
# env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
# env = VecNormalize.load(stats_path, env)


