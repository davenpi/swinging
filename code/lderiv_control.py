import copy
import gym
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


class Swing(gym.Env):
    """
    Swing environment.
    """

    def __init__(self):
        super(Swing, self).__init__()
        self.power_max = 2500  # number in Watts is max power output from muscles
        self.mass = 70  # 70 kg person
        self.lmin = 4  # 2
        self.lmax = 5  # 2.5
        self.phidot_0 = -0.1
        self.target = np.pi
        self.time = 0
        self.pumps = 0
        self.tau = np.sqrt(self.lmin / 9.81) / 4  # play with this
        self.ldot_max = (self.lmax - self.lmin) / (self.tau)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -10, self.lmin]),
            high=np.array([2 * np.pi, 10, self.lmax]),  # phi, phi dot, L
        )
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        self.phi = [np.pi / 10]
        self.phi_dot = [self.phidot_0]
        self.L = [self.lmin]
        self.Ldot_hist = []

    def fun(self, t, y, ldot, g=9.81):
        """Define system of equations to simulate"""
        y0_dot = y[1]
        y1_dot = -(2 * ldot / y[2]) * y[1] - (g / y[2]) * np.sin(y[0])
        y2_dot = ldot
        y_dot = np.hstack((y0_dot, y1_dot, y2_dot))
        return y_dot

    def forward(self, ldot):
        """Simulate the swing forward"""
        sol = si.solve_ivp(
            self.fun,
            [self.time, self.time + self.tau],
            y0=[self.phi[-1], self.phi_dot[-1], self.L[-1]],
            args=[ldot],
        )
        phi = np.mod(sol.y[0], 2 * np.pi)
        self.phi.extend(list(phi[1:]))
        phi_dot = sol.y[1]
        self.phi_dot.extend(list(phi_dot[1:]))
        L = sol.y[2]
        self.L.extend(list(L[1:]))
        self.time += self.tau
        self.pumps += 1

    def check_out_of_bounds_action(self, ldot):
        """Check if an action will take us out of bounds. if so don't allow it."""
        next_l = self.L[-1] + self.tau * ldot
        if next_l > self.lmax:
            ldot = (self.lmax - self.L[-1]) / self.tau
        elif next_l < self.lmin:
            ldot = (self.lmin - self.L[-1]) / self.tau
        else:
            ldot = ldot
        return ldot

    def check_max_power_action(self, ldot):
        power_bound = self.power_max / (
            (self.mass)
            * (9.81 * np.cos(self.phi[-1]) + self.L[-1] * self.phi_dot[-1] ** 2)
        )
        if ldot < power_bound:
            ldot = ldot
        else:
            ldot = power_bound
        return ldot

    def step(self, action):
        """Take action and simulate"""
        ldot = self.ldot_max * action[0]
        ldot = self.check_max_power_action(ldot)
        ldot = self.check_out_of_bounds_action(ldot)
        self.Ldot_hist.append(ldot)
        self.forward(ldot)
        state = np.array([self.phi[-1], self.phi_dot[-1], self.L[-1]], dtype=np.float32)
        if np.isclose(state[0], self.target, rtol=0.05):
            reward = 10
            done = True
        elif self.pumps > 2_000:
            reward = -1
            done = True
        else:
            reward = -1
            done = False
        info = {}
        reward -= (ldot**2) / self.ldot_max**2
        return state, reward, done, info

    def reset(self):
        """Reset system to beginning of episode."""
        self.time = 0
        self.pumps = 0
        self.L = [self.lmin]
        self.phi = [np.pi / 8]
        self.phi_dot = [self.phidot_0]
        self.Ldot_hist.clear()
        state = np.array([self.phi[-1], self.phi_dot[-1], self.L[-1]], dtype=np.float32)
        return state

    def render(self):
        pass
