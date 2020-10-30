"""two layers MLP"""
# import highway_env
# import gym
# import tensorflow as tf
#
# from stable_baselines import PPO2
#
# # custom MLP policy of two layers of size 32 each with tanh activation function
# policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
# # create the agent
# model = PPO2("MlpPolicy", "overtaking-v0", policy_kwargs=policy_kwargs, verbose=1)
# # retrieve the environment
# env = model.get_env()
# # train the agent
# model.learn(total_timesteps=1000)
# # save the agent
# model.save("overtaking-v0")
#
# del model
# # the policy_kwargs are automatically loaded
# model = PPO2.load("overtaking-v0")


"""three layers MLP"""
import gym
import highway_env

from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


# custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

# create and wrap the environment
env = gym.make("overtaking-v0")
env = DummyVecEnv([lambda: env])

model = A2C(CustomPolicy, env, verbose=1)
# train the agent
model.learn(total_timesteps=1000)
# save the agent
model.save("overtaking-v0")

del model
# when load a model with a custom policy
# you must pass explicitly the policy with loading the saved model
model = A2C.load("overtaking-v0", policy=CustomPolicy)
