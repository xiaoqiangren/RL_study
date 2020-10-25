import os
import gym
import numpy as np
import matplotlib.pyplot as plt
#import gym_environment
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC

from stable_baselines.common.env_checker import check_env, VecCheckNan
#from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy, FeedForwardPolicy
#from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import time

## these are used to customize cnn policies
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter

#disable tf warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)



SEED = 0
np.random.seed(SEED)

####  customise policy
############### policy with dense network

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,layers=[700,700,600],
                                           layer_norm=False,
                                           feature_extraction="mlp")

##########################
# customise callbacks
##########################
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
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

##########################
log_dir = "sac_logger/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('overtaking-v0')
env = Monitor(env, log_dir)

print("the action dimension:{}".format(env.action_space.shape[-1]))
#print(env.observation_space)

# It will check your custom environment and output additional warnings if needed
check_env(env)
#env = VecCheckNan(env, raise_exception=True)


# quick check with random actions on the env.
# Do this whenever a new environment is added on.
#obs = env.reset()
#n_steps = 10
#for _ in range(n_steps):
# # Random action
# action = env.action_space.sample()
# #print(action)
# obs, reward, done, infos = env.step(action)
# print("the reward we received:{}".format(reward))
# print("the action we received:{}".format(done))


model = SAC(CustomPolicy, env, gamma=0.9, learning_rate=0.008598549551828587, learning_starts=5, batch_size=64, buffer_size=1000000, train_freq=2,
            tau=0.01, ent_coef='auto',target_entropy=-100, verbose=1, tensorboard_log="./sac_filter_tensorboard/")

callback = SaveOnBestTrainingRewardCallback(check_freq=40, log_dir=log_dir)
time_steps = 3e4


start_time = time.time()
model.learn(total_timesteps=int(time_steps), callback=callback)
print("--- %s seconds ---" % (time.time() - start_time))

# save the last model
model.save("td3_lastmodel")

results_plotter.plot_results(
    [log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
plt.show()