import os

import gym
import highway_env
import pybullet_envs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines import PPO2


# save a model-best train reward
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
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
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


# callback tensorboard
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True
        # Log scalar value (here a random variable)
        value = np.random.random()
        summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


# create log dir
log_dir = "PPO2Overtaking_tmp/"
os.makedirs(log_dir, exist_ok=True)
# video_folder = '/home/cxc/下载/实验结果'
# video_length = 100

env = gym.make("overtaking-v0")
# env = DummyVecEnv([lambda: gym.make("overtaking-v0")])
# Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                    clip_obs=10.)
env = Monitor(env, log_dir)
# env = VecVideoRecorder(env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix='random-agent-{}'.format("overtaking-v0"))

model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./ppo2_overtaking_tensorboard/")
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="PPO2Overtaking_tmp/")
eval_env = env
eval_callback = EvalCallback(eval_env, best_model_save_path='PPO2Overtaking_tmp/best_model',
                             log_path="PPO2Overtaking_tmp/results", eval_freq=500)
callback = SaveOnBestTrainingRewardCallback(check_freq=200, log_dir=log_dir)
callbacks = CallbackList([callback, checkpoint_callback, eval_callback])
time_steps = 5000
model.learn(total_timesteps=int(time_steps), callback=callbacks)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, " PPO2 OvertakingEnv ")
plt.show()



# model.learn(total_timesteps=2000,callback=TensorboardCallback())
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()

# #  do not update them at test time
# env.training = False
# # reward normalization is not needed at test time
# env.norm_reward = False

# Don't forget to save the VecNormalize statistics when saving the agent
# llog_dir = "/home/cxc/下载/实验结果"
# model.save(llog_dir + "ppo_overtaking")
# stats_path = os.path.join(llog_dir, "vec_normalize.pkl")
# env.save(stats_path)
#
# # To demonstrate loading
# del model, env
#
# # Load the agent
# model = PPO2.load(log_dir + "ppo_overtaking")



# Load the saved statistics
# env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
# env = VecNormalize.load(stats_path, env)


