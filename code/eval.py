import non_dim_lderiv_control as ld
from stable_baselines3 import PPO, SAC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

env = ld.Swing()
lmin = 0.9
lmax = 1.1
phi_0 = np.pi / 8
phidot_0 = 0
tau = 0.25  # (lmax - lmin) / 4
ldot_max = 0.25
env.ldot_max = ldot_max
env.lmin = lmin
env.lmax = lmax
env.ldot_max = ldot_max
env.phi_0 = phi_0
env.phidot_0 = phidot_0
env.tau = tau
env.power_max = 0.25

model = SAC.load("w_power_logs/rl_model_300000_steps", env=env)


done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    # if env.pumps > 100:
    #     break

phi_hist = np.array(env.phi)
l_hist = np.array(env.L)

x_t = l_hist * np.sin(phi_hist)
y_t = -l_hist * np.cos(phi_hist)
ref_x = np.array(env.lmax) * np.sin(env.phi)
ref_y = -np.array(env.lmax) * np.cos(env.phi)

bound = 1.2 * env.lmax


def animate(i):
    ax.clear()
    ax.plot([0, ref_x[i]], [0, ref_y[i]])
    ax.plot([0, x_t[i]], [0, y_t[i]], "o")
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_title("Swinging over time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig, ax = plt.subplots(figsize=(10, 10))
# run the animation
ani = FuncAnimation(fig, animate, frames=x_t.size, interval=10, repeat=False)

writervideo = animation.FFMpegWriter(fps=8)
ani.save("video.mp4", writer=writervideo)
plt.close()
