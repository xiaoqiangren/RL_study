""" use dqn to train the overtaking environment"""
import os

import gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# from rl_agents.agents.common.models import EgoAttentionNetwork


# save a model
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


# create log dir
log_dir = "DqnOvertaking_tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('overtaking-v0')
env = Monitor(env, log_dir)

# Instantiate the agent
# model = DQN(EgoAttentionNetwork, env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model = DQN("MlpPolicy", env, learning_rate=1e-3, prioritized_replay=True, verbose=1,
            tensorboard_log="./DQN_overtaking_tensorboard/")
# create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 1e5
model.learn(total_timesteps=int(time_steps), callback=callback)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DQN OvertakingEnv")
plt.show()

# # Save the agent
# model.save("dqn_overtaking")
# del model  # delete trained model to demonstrate loading
#
# # Load the trained agent
# model = DQN.load("dqn_overtaking", env=env)
#
# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# obs = env.reset()
# episode_reward = 0
# for i in range(400):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
#     # episode_reward += reward
#     # if done or info.get('is_success', False):
#     #     print("Reward:", episode_reward, "Success?", info.get('is_success', False))
#     #     episode_reward = 0.0
#     #     obs = env.reset()
