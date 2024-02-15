"""
Environment to calculate the Whittle index values as a deep reinforcement
learning environment modelled after the OpenAi Gym API.
From the paper:
"Restless Bandits with Controlled Restarts"
"""
import gym
import random
import numpy as np
from gym import spaces
from MarkovEnv import Reward, MarkovDynamics


class RestartBanditEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment modelled after "deadline scheduling as restless bandits" 
    paper RMAB description. The environment represents one position in the N-length queue. 
    '''

    def __init__(self, seed, numEpisodes, episodeLimit, num_states, function_type, reset_coef,
                 prob_remain, transition_type, discount, cost_threshold, train, batchSize, noiseVar):
        super(RestartBanditEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed)  # create a special PRNG for a class instantiation

        self.observationSize = 2
        self.threshold = cost_threshold

        # first: state. Second: threshold_state
        self.arm = np.array([0, self.threshold], dtype=np.float32)

        self.noiseVar = noiseVar
        self.numEpisodes = numEpisodes
        self.currentEpisode = 0
        self.episodeTime = 0
        self.episodeLimit = episodeLimit
        self.train = train

        self.num_states = num_states
        R = Reward(1, num_states, function_type, reset_coef)
        self.cost_set = R.costs[:, :, 0]
        M = MarkovDynamics(1, num_states, prob_remain, transition_type)
        self.transitions = M.transitions[:, :, :, 0]
        self.discount = discount

        self.batchSize = batchSize
        self.miniBatchCounter = 0
        self.loadIndex = 0

        self.action_space = spaces.Discrete(2)

        self.load = None

    def _calRewardAndState(self, action):
        """ function to calculate the reward and next state. """
        immediate_cost = (1 - self.discount) * self.cost_set[int(self.arm[0]), action]

        transition_prob = self.transitions[int(self.arm[0]), :, action]
        self.arm[0] = np.random.choice(np.arange(len(transition_prob)), p=transition_prob)
        print(self.arm[1])
        print(immediate_cost)
        print(self.discount)
        self.arm[1] = max(0, (self.arm[1] - immediate_cost) / self.discount)
        print(self.arm[1])
        print('----------')

        return np.array([self.arm[0], self.arm[1]], dtype=np.float32), 0

    def step(self, action):
        """ standard Gym function for taking an action. Provides the next state, reward,
        and episode termination signal. """
        assert self.action_space.contains(action)
        assert action in [0, 1]
        self.episodeTime += 1

        next_state, cost = self._calRewardAndState(action)

        if self.train:
            done = bool(self.episodeTime == self.episodeLimit)
        else:
            done = False

        if done:
            self.currentEpisode += 1
            self.episodeTime = 0
            if not self.train:
                self.currentEpisode = 0

        return next_state, cost, done, {}

    def reset(self):
        """ standard Gym function for reseting the state for a new episode."""
        self.loadIndex = 0

        if self.miniBatchCounter % self.batchSize == 0:
            self.miniBatchCounter = 0

        self.episodeTime = 0
        self.loadIndex += 1
        self.miniBatchCounter += 1

        return np.array([0, self.threshold], dtype=np.float32)


#########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 50

env = deadlineSchedulingEnv(seed = SEED, numEpisodes = 6, episodeLimit = 20, maxDeadline = 12,
maxLoad=9, newJobProb=0.7, train=True, processingCost = 0.5, batchSize = 1, noiseVar = 0.0)

observation = env.reset()

#check_env(env, warn=True)

x = np.array([1,1,0,0,1])
x = np.tile(x, 10000)
#x = np.random.choice([1,0], size=1000)
n_steps = np.size(x)

start = time.time()
for step in range(n_steps):
    nextState, reward, done, info = env.step(x[step])
    print(f'action: {x[step]} nextstate: {nextState}  reward: {reward} done: {done}')
    print("---------------------------------------------------------")
    if done:
        print(f'Finished episode {env.currentEpisode}/{env.numEpisodes}')
        if env.currentEpisode < env.numEpisodes:
            nextState = env.reset()
        if env.currentEpisode == env.numEpisodes:
            break
  

print(f'-------------------------------------\nDone. Time taken: {time.time() - start:.4f} seconds')
'''
