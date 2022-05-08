from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

fig, ax = plt.subplots()
pos_arr = np.load("nbs/pos_arr.npy")


def animate(i):
    ax.clear()

    ax.plot([0, pos_arr[0][i]], [0, pos_arr[1][i]], "o-")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_title("Swinging over time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# run the animation
ani = FuncAnimation(fig, animate, frames=pos_arr[0].size, interval=500, repeat=False)

writervideo = animation.FFMpegWriter(fps=3)
ani.save("swing_vid.mp4", writer=writervideo)
plt.close()
