import gym
import highway_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2


# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = gym.make("overtaking-v0")
env = DummyVecEnv([lambda: env])
# check_env(env)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo2_overtaking_tensorboard/")
model.learn(total_timesteps=10000)

obs = env.reset()
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()

