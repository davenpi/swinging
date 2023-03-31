import non_dim_lderiv_control as ld
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


lmin = 0.95
lmax = 1.05
phi_0 = np.pi / 4
phidot_0 = 0
tau = 0.1  # (lmax - lmin) / 4
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
    save_path="bounded_models/",
    name_prefix="rl_model",
)

model_path = "bounded_final_trained_w_energy_ratio/rl_model_1000000_steps.zip"
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="unbounded_logs/")
model = PPO.load(
    model_path,
    env=env,
    tensorboard_log="bounded_logs/",
)
model.learn(
    total_timesteps=6e5,
    callback=checkpoint_callback,
    progress_bar=True,
)
