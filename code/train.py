import non_dim_lderiv_control as ld

# import energy_reward as ld
import copy
import gym
import numpy as np
import pandas as pd
import scipy.integrate as si
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from stable_baselines3 import A2C, PPO, SAC, TD3, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback


lmin = 0.95
lmax = 1.05
phi_0 = np.pi / 4
phidot_0 = 0
tau = 0.125  # (lmax - lmin) / 4
ldot_max = 0.1
power_max = 0.100

power_bounded = power_max < 1
env = ld.Swing(power_bounded=power_bounded)
env.ldot_max = ldot_max
env.lmin = lmin
env.lmax = lmax
env.ldot_max = ldot_max
env.phi_0 = phi_0
env.phidot_0 = phidot_0
env.tau = tau
env.power_max = power_max

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./big_state_unbounded_models/",
    name_prefix="rl_model",
)

# policy_kwargs = dict(net_arch=dict(pi=[256, 256]))
model = SAC("MlpPolicy", env, verbose=0, tensorboard_log="big_state_unbounded_logs/")
# model = SAC.load("big_state_unbounded_models/rl_model_300000_steps.zip", env = env, tensorboard_log="big_state_unbounded_logs/")
model.learn(total_timesteps=6e5, callback=checkpoint_callback)
