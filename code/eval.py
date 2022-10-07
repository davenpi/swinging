import lderiv_control as ld
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

env = ld.Swing()
# model = PPO.load("trained_model_new.zip", env=env)
model = PPO.load("logs/rl_model_10000_steps.zip", env=env)


done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

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
ani = FuncAnimation(fig, animate, frames=x_t.size, interval=100, repeat=False)

writervideo = animation.FFMpegWriter(fps=8)
ani.save("ani_vid_10000_steps.mp4", writer=writervideo)
plt.close()
