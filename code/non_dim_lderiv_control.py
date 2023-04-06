"""
This script implement the swing as an OpenAI gym environment. With the system
as a gym environment we can use any off the shelf reinforcement learning
algorithms to try and optimize performance by maximizing accumulated reward.
"""
import gym
import numpy as np
import scipy.integrate as si


class Swing(gym.Env):
    """
    Swing environment.
    """

    def __init__(self, power_bounded: bool):
        super(Swing, self).__init__()
        self.power_max = 0.1
        self.lmin = 0.9
        self.lmax = 1.1
        self.target = np.pi
        self.time = 0
        self.pumps = 0
        self.tau = 0.1  # simulation time step
        self.ldot_max = 0.1
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, 0.5]),
            high=np.array([10, 10, 2]),  # phi, p = l^2*phi_dot, l
        )
        self.action_space = gym.spaces.Discrete(3)
        self.phi_0 = np.pi / 4
        self.phi = [self.phi_0]
        self.phidot_0 = 0.0
        self.phi_dot = [self.phidot_0]
        self.L = [self.lmax]
        self.Ldot_hist = [0]
        self.pump_limit = 6000
        self.rtol = 0.02
        self.power_bounded = power_bounded
        # gym discrete actions are 0, ..., n. We want the actions to be
        # {0, 1, -1}. This dictionary performs the remapping of the actions.
        self.discrete_action_lookup = {0: 0, 1: 1, 2: -1}

    def dynamics(self, t: float, y: np.ndarray, ldot: float) -> np.ndarray:
        """
        Define the system of equations to simulate and compute derivatives.

        This function implements the dynamical equations governing the system.
        The if/else logic checks whether or not we are solving the power
        bounded version of the problem. If so, we need to make sure the current
        value of the control gets modified so that it falls within the power
        bounds (eq. 4 in the accompanying paper). Returns the derivatives of
        each of the variables to simulate the system forward in time.

        Parameters
        ----------
        t : float
            Instantaneous time.
        y : np.ndarray
            Array containing (theta, omega, l) values at a given time. Not the
            same as the state.
        ldot : float
            Value of the control variable u = ldot at a given time.

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
        y1_dot = -(2 * ldot / y[2]) * y[1] - (1 / y[2]) * np.sin(y[0])
        y2_dot = ldot
        y_dot = np.hstack((y0_dot, y1_dot, y2_dot))
        return y_dot

    def forward(self, ldot: float) -> None:
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
        None
        """
        sol = si.solve_ivp(
            self.dynamics,
            [self.time, self.time + self.tau],
            y0=[self.phi[-1], self.phi_dot[-1], self.L[-1]],
            args=[ldot],
        )
        phi = sol.y[0]
        self.phi.extend(list(phi[-1:]))
        phi_dot = sol.y[1]
        self.phi_dot.extend(list(phi_dot[-1:]))
        L = sol.y[2]
        self.L.extend(list(L[-1:]))
        self.time += self.tau
        self.pumps += 1

    def check_out_of_bounds_action(self, ldot: float) -> float:
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

    def check_max_power_action(self, ldot: float) -> float:
        """
        Check if the suggested by the policy is within the power bound.

        Since we are limiting the amount of power the person can output, we
        need to make sure that the actions taken by the agent are within the
        specified bounds.

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

    def compute_energy(self) -> float:
        """
        Compute the sum of the potential and kinetic energy of the swing.

        Parameters
        ----------
        None

        Returns
        -------
        energy : float
            Sum of potential and kinetic energy.
        """
        KE = 0.5 * (self.phi_dot[-1] * self.L[-1]) ** 2
        PE = -self.L[-1] * np.cos(self.phi[-1])
        energy = KE + PE
        return energy

    def extract_state(self) -> np.ndarray:
        """
        Extract the state of the system.

        The state is given by (theta, p, l) and matches the control theory.
        Here p = omega*l^2 is the angular momentum.

        Parameters
        ----------
        None

        Returns
        -------
        state : np.ndarray
            1d array with state (theta, p, l)
        """
        theta = self.phi[-1]
        p = self.L[-1] * self.phi_dot[-1] ** 2
        l = self.L[-1]
        state = np.array([theta, p, l], dtype=np.float32)
        return state

    def step(self, action: int) -> tuple:
        """
        Step is a method required by all gym envs. Advance the system in time.

        Here we move the simulation forward in time with the given action.
        The reward is structured so that every time step the agent gets a
        positive reward for its energy and a negative reward for not yet
        reaching the goal state. The if/else logic inside checks whether or
        not the agent is sufficiently close to the goal state on the most
        recent time steps. Each simulation interval has intermediate values
        and we check whether or not the agent reached the goal at one of the
        intermediate values between time steps. This checking ensures that we
        end the episode properly. During training we set rtol, the relative
        tolerance/allowed percentage deviating from the true final state, to
        2 percent. After training and during evaluation of the agent we set
        rtol to be 0.1 percent.

        Parameters
        ----------
        action : np.ndarray
            Action specified by agent. This is a positive or negative
            fraction of the maximum control allowable.

        Returns
        -------
        state : np.ndarray
            State of the system (angle, angular momentum, length). This is
            the same state as described in equation 1 of the accompanying
            paper.
        done : Bool
            True if episode is over because the agent accomplished the task or
            ran out of time. False otherwise.
        reward : float
            Feedback signal given to the agent to help with learning. The
            accumulated reward is what the agent tries to maximize over time.
        """
        action = self.discrete_action_lookup[action]
        ldot = self.ldot_max * action
        ldot = self.check_out_of_bounds_action(ldot)
        if self.power_bounded:
            ldot = self.check_max_power_action(ldot)
        self.Ldot_hist.append(ldot)
        self.forward(ldot)
        energy = self.compute_energy()
        energy_ratio = energy / self.L[-1]
        mod_phi = np.mod(self.phi[-1], 2 * np.pi)
        mod_phi_prev = np.mod(self.phi[-2], 2 * np.pi)
        if np.isclose(mod_phi, self.target, rtol=self.rtol) or np.isclose(
            mod_phi_prev, self.target, rtol=self.rtol
        ):
            reward = 1
            done = True
        elif self.pumps > self.pump_limit:
            reward = -1
            done = True
        else:
            reward = -1 + energy_ratio
            done = False
        info = {}
        state = self.extract_state()
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset system to beginning of episode.

        I am restarting the agent from the same initial state every time. This
        method also clears out all of the accumulated history from the
        most recent training run.

        Parameters
        ----------
        None

        Returns
        -------
        state : np.ndarray
            State of the system at the initial time.
        """
        self.time = 0
        self.pumps = 0
        self.L = [self.lmax]
        self.phi = [self.phi_0]
        self.phi_dot = [self.phidot_0]
        self.Ldot_hist = [0]
        state = self.extract_state()
        return state

    def render(self):
        """
        Required method for gym environments. Used to visualize the system.

        Not necessary to implement so I have left if blank.
        """
        pass
