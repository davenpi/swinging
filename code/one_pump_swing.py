import gym
import numpy as np
import scipy.integrate as si


class Swing(gym.Env):
    """
    Swing environment.
    """

    def __init__(self, power_bounded: bool):
        super(Swing, self).__init__()
        self.power_max = 1000  # number in Watts is max power output from muscles
        self.mass = 70  # 70 kg person
        self.lmin = 4.5  # minimum pendulum length in meters
        self.lmax = 5.5  # maximum pendulum length in meters. assumed human is 1.8 m
        self.target = np.pi
        self.time = 0
        self.pumps = 0
        self.tau = np.sqrt(self.lmin) / 4
        self.ldot_max = (self.lmax - self.lmin) / (2 * self.tau)
        # self.observation_space = gym.spaces.Box(
        #     low=np.array([0, -10, self.lmin]),
        #     high=np.array([2 * np.pi, 10, self.lmax]),  # phi, phi dot, L
        # )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -10, self.lmin]),
            high=np.array([1, 1, 10, self.lmax]),  # cos(phi), sin(phi), phi dot, L
        )
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        self.phi_0 = np.pi / 2
        self.phi = [self.phi_0]
        self.phidot_0 = 0.0
        self.phi_dot = [self.phidot_0]
        self.L = [self.lmax]
        self.Ldot_hist = [0]
        self.pump_limit = 5000
        self.rtol = 0.05
        self.power_bounded = power_bounded
        self.discrete_action_lookup = {0: 0, 1: 1, 2: -1}

    def fun(self, t, y, ldot, g=9.81):
        """
        Define system of equations to simulate.

        Parameters
        ----------
        t : float
            Instantaneous time. Used if equation depends explicitly on time.
        y : np.ndarray
            Array containing (theta, omega, l) values at a given time.
        ldot : float
            Value of the control variable u = ldot at a given time.
        g : float
            Gravitational constant.

        Returns
        -------
        y_dot : np.ndarray
        """
        if self.power_bounded == True:
            power_bounded_u = np.abs(self.power_max / (y[2] * y[1] ** 2 + np.cos(y[0])))
            if np.abs(ldot) < power_bounded_u:
                ldot = ldot
            else:
                if ldot > 0:
                    ldot = power_bounded_u
                else:
                    ldot = -power_bounded_u
        else:
            ldot = ldot

        y0_dot = y[1]
        # y1_dot = -(2 * ldot / y[2]) * y[1] - (g / y[2]) * np.sin(y[0])
        y1_dot = -(2 * ldot / y[2]) * y[1] - (1 / y[2]) * np.sin(
            y[0]
        )  # non dimensional version
        y2_dot = ldot
        y_dot = np.hstack((y0_dot, y1_dot, y2_dot))
        return y_dot

    def forward(self, ldot):
        """
        Simulate the swing forward in time by one time step.

        Here we simulate the differential equations describing the system
        forward in time by one time step. After simulating the equations
        I append solutions to the state history. I also increment time
        forward by the simulation timestep "tau" and by incrementing the
        total number of pumps by one.

        Parameters
        ----------
        ldot : float
            Value of control at current time step.

        Returns
        -------
        """

        sol = si.solve_ivp(
            self.fun,
            [self.time, self.time + self.tau],
            y0=[self.phi[-1], self.phi_dot[-1], self.L[-1]],
            args=[ldot],
        )
        phi = np.mod(sol.y[0], 2 * np.pi)
        self.phi.extend(list(phi[-1:]))
        phi_dot = sol.y[1]
        self.phi_dot.extend(list(phi_dot[-1:]))
        L = sol.y[2]
        self.L.extend(list(L[-1:]))
        self.time += self.tau
        self.pumps += 1

    def check_out_of_bounds_action(self, ldot):
        """
        Check if an action will take the length out of allowed range.

        If the action will take the length out of the allowed range then
        change the magnitude of the action to ensure we stay within the bounds.

        Parameters
        ----------
        ldot : float
            Magnitude of the control.

        Returns
        -------
        ldot : float
            Same as the input action if that is valid. Otherwise we change
            the magnitude so the action takes us to but not beyond the bound
            on length.
        """
        next_l = self.L[-1] + self.tau * ldot
        if next_l > self.lmax:
            ldot = (self.lmax - self.L[-1]) / self.tau
        elif next_l < self.lmin:
            ldot = (self.lmin - self.L[-1]) / self.tau
        else:
            ldot = ldot
        return ldot

    def check_max_power_action(self, ldot):
        """
        Check if the suggested by the policy is within the power bound.

        Since we are limiting the amount of power the person can output, we
        need to make sure that the actions taken by the agent are within the
        specified bounds. The trick is that the value of the control which is
        within the power bounds depends on the state of the system. So first
        we compute the state dependent power bound formula that is derived in
        the paper. After we have computed that bound we then check to see if
        the action picked by the RL agent in within that bound. If it is,
        do nothing. If the action is outside of the bound, set the new action
        to be the maximum power that can be supplied. Note this should only be
        applied for actions where ldot is less than zero (ldot < 0)becasue
        those are the cases where the swinger wants to stand up and therefore
        use effort. For now we can assume that squatting down requires no
        power output.

        Parameters
        ----------
        ldot : float
            Action suggested by policy.

        Returns
        -------
        ldot : float
            Same as policy action if the suggested action is within bound.
            Otherwise return the maximum power action allowed in this state.
        """
        # power_bounded_u = np.abs(
        #     self.power_max
        #     / (
        #         (self.mass)
        #         * (
        #             -9.81 * (1 - np.cos(self.phi[-1]))
        #             + self.L[-1] * self.phi_dot[-1] ** 2
        #         )
        #     )
        # )

        power_bounded_u = np.abs(
            self.power_max / (self.L[-1] * self.phi_dot[-1] ** 2 + np.cos(self.phi[-1]))
        )
        if np.abs(ldot) < power_bounded_u:
            ldot = ldot
        else:
            if ldot > 0:
                ldot = power_bounded_u
            else:
                ldot = -power_bounded_u
        return ldot

    def action_rounder(self, action):
        if action > 0.5:
            action = 1
        elif action < -0.5:
            action = -1
        else:
            action = 0
        return action

    def step(self, action):
        """
        Implementing the step method required of all gym environments.

        Here we move the simulation forward in time with the given action.
        The reward is structured so that every time step is penalized i.e we
        want the agent to complete the task as quickly as possbile. We have a
        small reward for getting to phi = pi. The reward for swinging up to
        phi = pi is actually not quite what I want. I want the agent to
        get as high as possible as quickly as possible. I NEED TO FIX THAT.
        There is an end condition which says that if the agent has
        tried 5000 pumps up and down and is yet to solve the task, end the
        episode. Finally there is some reward associated with using less the
        amount of effort the agent is using to complete the task. The effort
        dependent reward is normalized and the weighting factor determines
        how much we care about using energy vs taking time to complete the
        task.
        I return the state (angle, angular velocity, length), reward,
        whether or not the episode is over, and some optional logging info
        after every time step.

        Parameters
        ----------
        action : np.ndarray
            Action specified by agent. This is a positive or negative
            fraction of the maximum control allowable.

        Returns
        -------
        state : np.ndarray
            State of the system (angle, angular velcoity, length). This is
            a choice and experimenting with different states is cetnratinly
            on the table.
        done : Bool
            True if episode is over because the agent accomplished the task or
            ran out of time. False otherwise.
        reward : float
        """

        action = self.action_rounder(action[0])
        ldot = self.ldot_max * action
        ldot = self.check_max_power_action(ldot)
        ldot = self.check_out_of_bounds_action(ldot)
        self.Ldot_hist.append(ldot)
        self.forward(ldot)
        state = np.array(
            [np.cos(self.phi[-1]), np.sin(self.phi[-1]), self.phi_dot[-1], self.L[-1]],
            dtype=np.float32,
        )
        recent_phi_dots = self.phi_dot[
            -2:
        ]  # not perfect. could miss a turnaround that happens and then both omega are negative.
        if np.prod(recent_phi_dots) <= 0:
            reward = np.abs(self.phi[-1])
            done = True
        elif self.pumps > self.pump_limit:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        """
        Reset system to beginning of episode.

        I am restarting the agent from the same initial state every time. This
        method is here to clear out all of the accumulated history from the
        most recent training run.
        """
        self.time = 0
        self.pumps = 0
        self.L = [self.lmax]
        rand_num = np.random.random(1)[0]
        self.phi = [rand_num * self.phi_0]
        self.phi_dot = [self.phidot_0]
        self.Ldot_hist = [0]
        state = np.array(
            [np.cos(self.phi[-1]), np.sin(self.phi[-1]), self.phi_dot[-1], self.L[-1]],
            dtype=np.float32,
        )
        return state

    def render(self):
        pass
