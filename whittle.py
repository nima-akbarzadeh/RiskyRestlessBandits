import numpy as np


class Whittle:

    def __init__(self, num_states: int, num_arms: int, num_augmented: int, cost, transition, horizon, beta=None):
        self.num_x = num_states
        self.num_a = num_arms
        self.num_s = num_augmented
        self.cost = cost
        self.transition = transition
        self.horizon = horizon
        self.beta = beta

    def get_whittle_indices(self, computation_type):
        if computation_type == 1:
            whittle_indices = self.whittle_policy_search()
        elif computation_type == 2:
            whittle_indices = self.whittle_binary_search()
        else:
            whittle_indices = self.whittle_brute_force()
        return whittle_indices

    def whittle_policy_search(self):
        w_indices = np.zeros([self.num_x, self.num_s, self.num_a])
        return w_indices

    def whittle_binary_search(self):
        w_indices = np.zeros([self.num_s, self.num_a])
        return w_indices

    def whittle_brute_force(self):
        w_indices = np.zeros([self.num_s, self.num_a])
        return w_indices

    ######## 4 Multiplies
    def backward_induction_augstate(self, penalty, tau, eps):

        # Set L to be an odd number
        assert np.mod(self.num_s, 4) == 1

        # Value function initialization
        V = np.zeros([self.num_s, self.num_x, self.horizon+1], dtype=np.float32)

        # Half index (4)
        HL = int(0.5 * (self.num_s - 1))

        # initial index (2)
        IL = int(0.25 * (self.num_s - 1))

        # final index (6)
        FL = int(0.75 * (self.num_s - 1))

        # Maximum total cost
        maxd_cost = 2 * self.horizon

        # Identify the value function at T+1
        t = self.horizon

        # Loop over the first dimension of the state space
        for x in range(self.num_x):

            # Loop over the second dimension of the state space
            for s in range(IL, FL+1):

                # Convert indices into total cost values ranged from -T to T
                cur_level = (-1 + s / HL) * maxd_cost
                V[s, x, t] = np.maximum(0, -cur_level)

        # State-action value function
        Q = np.zeros([self.num_s, self.num_x, self.horizon, 2], dtype=np.float32)

        # Policy function
        pi = np.zeros([self.num_s, self.num_x, self.horizon], dtype=np.int32)

        # Backward induction timing
        t = self.horizon-1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for s in range(self.num_s):

                    # Convert indices into total cost values ranged from -T to T
                    cur_level = (-1 + s / HL) * maxd_cost

                    # Get the next state in the second dimension
                    next_stat = cur_level - self.cost[x]

                    # Convert the next state of the second dimension into an index ranged from 1 to L
                    next_s = int(max(0, min(self.num_s - 1, int(round(HL * (1 + (next_stat / maxd_cost)))))))

                    # Get the state-action value functions
                    Q[s, x, t, 0] = np.dot(V[next_s, :, t+1], self.transition[x, :, 0])
                    Q[s, x, t, 1] = penalty + np.dot(V[next_s, :, t+1], self.transition[x, :, 1])

                    # Get the value function and the policy
                    if Q[s, x, t, 0] < Q[s, x, t, 1]:
                        V[s, x, t] = Q[s, x, t, 0]
                        pi[s, x, t] = 0
                    else:
                        V[s, x, t] = Q[s, x, t, 1]
                        pi[s, x, t] = 1

            t = t - 1

        return pi, V, Q


if __name__ == '__main__':

    from envs.MarkovEnv import MarkovDynamics, Reward
    horizon = 10

    num_arms = 1
    num_states = 3
    num_augmented = 4 * num_states * horizon + 1
    function_type = 'linear'
    reset_coef = 1.5
    prob_remain = 0.5 * np.ones(num_arms)
    transition_type = 1
    R = Reward(1, num_states, function_type, reset_coef)
    M = MarkovDynamics(1, num_states, prob_remain, transition_type)

    horizon = 20
    W = Whittle(num_states, num_arms, num_augmented, R.costs[:, 0, 0], M.transitions[:, :, :, 0], horizon)

    penalty = 1
    tau = 10
    eps = 1e-5

    pi, V, Q = W.backward_induction_augstate(penalty, tau, eps)

    print(pi)
    print(V)
    print(Q)
