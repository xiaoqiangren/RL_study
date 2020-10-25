import os

import gym
import numpy as np
import highway_env
import pprint
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env, VecCheckNan
from stable_baselines import PPO2
import time

# these are used to customize cnn policies
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter

# disable tf warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any{'0','1','2'}
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging

tf.get_logger().setLevel(logging.ERROR)

SEED = 0
np.random.seed(SEED)


# customise policy
# policy with dense network
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, laywers=[700, 700, 600],
                                           layer_norm=False,
                                           feature_extraction="mlp")


# customise callbacks
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
       Callback for saving a model (the check is done every ``check_freq`` steps)
       based on the training reward (in practice, we recommend using ``EvalCallback``).

       :param check_freq: (int)
       :param log_dir: (str) Path to the folder where the model will be saved.
         It must contains the file created by the ``Monitor`` wrapper.
       :param verbose: (int)
       """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 40 episodes
                mean_reward = np.mean(y[-40:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


log_dir = "pp02_logger/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make("overtaking-v0")
env = Monitor(env, log_dir)

# print("the action dimension:{}".format(env.action_space.shape[-1]))
# print(env.observation_space)

# it will check your custom environment and output additional warning if needed
check_env(env)
#env = VecCheckNan(env, raise_exception=True)

# quick check with random actions on the env.
# do this whenever a new environment is add on.
obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # random action
    # action = env.action_space.sample()
    action = env.action_type.actions_indexes["IDLE"]
    # obs, reward, done, info = env.step(action)
    # print(action)
    obs, reward, done, infos = env.step(action)
    print("the reward we received:{}".format(reward))
    print("the action we received:{}".format(action))

# #self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
#                  max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
#                  verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
#                  full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
model = PPO2(CustomPolicy, env, gamma=0.9, learning_rate=0.0086, nminibatches=64,
             verbose=1, tensorboard_log="./ppo2_filiter_tensorboard/")

callback = SaveOnBestTrainingRewardCallback(check_freq=40,log_dir=log_dir)
time_steps = 3e4

start_time = time.time()
model.learn(total_timesteps=int(time_steps),callback=callback)
"---%s seconds ---" % (time.time() - start_time)

# save the last model
model.save("ppo2_lastmodel")


results_plotter.plot_results([log_dir],time_steps,results_plotter.X_TIMESTEPS,"PPO2 LunarLander")
plt.show()


# done = False
# while not done:
#     #action = ... # Your agent code here
#     obs, reward, done, info = env.step(env.action_sapce.sample())
#     env.render()

# for _ in range(10):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(env.action_space.sample())
#     env.render()

# env = gym.make("overtaking-v0")
# env.reset()
# done = False
# while not done:
#     #action = ... # Your agent code here
#     obs, reward, done, info = env.step(env.action_sapce.sample())
#     env.render()

# env = gym.make('overtaking-v0')
# # Optional: PPO2 requires a vectorized environment to run
# # the env is now wrapped automatically when passing it to the constructor
# # env = DummyVecEnv([lambda: env])
#
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
#
# obs = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
#
# env.close()

# env = gym.make('merge-v0') ##创建高速公路模型
# pprint.pprint(env.config)
# env.configure({
#     'manual_control':False
# })
# env.reset()                 ##初始化环境
# for _ in range(5):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()           ##刷新当前环境并显示
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()

# env = gym.make('overtaking-v0')
# obs = env.reset()
# for _ in range(1000):
#   obs, reward, done, info = env.step(env.action_space.sample())
#   env.render()
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
