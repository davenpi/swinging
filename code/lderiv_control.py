import gym
import numpy as np
import scipy.integrate as si


class Swing(gym.Env):
    """
    Swing environment.
    """

    def __init__(self):
        super(Swing, self).__init__()
        self.power_max = 1000  # number in Watts is max power output from muscles
        self.mass = 70  # 70 kg person
        self.lmin = 4.25  # minimum pendulum length in meters
        self.lmax = 5.75  # maximum pendulum length in meters. assumed human is 1.8 m
        self.phidot_0 = 0.0614
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
        self.phi = [5.72]
        self.phi_dot = [self.phidot_0]
        self.L = [self.lmin]
        self.Ldot_hist = [0]

    def fun(self, t, y, ldot, g=9.81):
        """
        Define system of equations to simulate.

        Parameters
        ----------
        t : float
            Instantaneous time. Used if equation depends explicitly on time.
        y : np.ndarray
            Array containing state values at a given time.
        ldot : float
            Value of the control variable u = ldot at a given time.
        g : float
            Gravitational constant.

        Returns
        -------
        y_dot : np.ndarray
        """
        y0_dot = y[1]
        y1_dot = -(2 * ldot / y[2]) * y[1] - (g / y[2]) * np.sin(y[0])
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
        self.phi.extend(list(phi[1:]))
        phi_dot = sol.y[1]
        self.phi_dot.extend(list(phi_dot[1:]))
        L = sol.y[2]
        self.L.extend(list(L[1:]))
        self.time += self.tau
        self.pumps += 1
        # self.get_max_u()

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
        if ldot > 0:
            return ldot
        else:
            power_bounded_u = np.abs(
                self.power_max
                / (
                    (self.mass)
                    * (
                        -9.81 * (1 - np.cos(self.phi[-1]))
                        + self.L[-1] * self.phi_dot[-1] ** 2
                    )
                )
            )
            if np.abs(ldot) < power_bounded_u:
                ldot = ldot
            else:
                ldot = -power_bounded_u
            return ldot

    # def check_if_rotated_above_pi_from_left(self):
    #     phi_curr = self.phi[-1]
    #     phi_past = self.phi[-2]
    #     # rotated past pi from left to right
    #     if (
    #         (phi_curr < phi_past)
    #         and phi_curr <= np.pi
    #         and np.isclose(phi_curr, np.pi, rtol=0.05)
    #     ):  # check we are still rotatin up to end it
    #         done = True
    #     else:
    #         done = False
    #     return done

    # def check_if_rotated_above_pi_from_right(self):
    #     phi_curr = self.phi[-1]
    #     phi_past = self.phi[-2]
    #     # rotated past pi from left to right
    #     if (
    #         (phi_curr > phi_past)
    #         and phi_curr >= np.pi
    #         and np.isclose(phi_curr, np.pi, rtol=0.05)
    #     ):  # check we are still rotatin up to end it
    #         done = True
    #     else:
    #         done = False
    #     return done

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
        ldot = self.ldot_max * action[0]
        ldot = self.check_max_power_action(ldot)
        ldot = self.check_out_of_bounds_action(ldot)
        self.Ldot_hist.append(ldot)
        self.forward(ldot)
        state = np.array([self.phi[-1], self.phi_dot[-1], self.L[-1]], dtype=np.float32)
        phi_prev = self.phi[-2]
        if np.isclose(state[0], self.target, rtol=0.05) or np.isclose(
            phi_prev, self.target, rtol=0.05
        ):
            reward = 1
            done = True
        elif self.pumps > 5_000:
            reward = -1
            done = True
        else:
            reward = -1
            done = False
        info = {}
        reward -= 2 * (ldot**2) / self.ldot_max**2  # worked with multiply by 2
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
        self.L = [self.lmin]
        self.phi = [5.72]
        self.phi_dot = [self.phidot_0]
        self.Ldot_hist = [0]
        state = np.array([self.phi[-1], self.phi_dot[-1], self.L[-1]], dtype=np.float32)
        return state

    def render(self):
        pass
