"""
NEURWIN algorithm: used for learning
the Whittle index of one restless arm.
Training is done in a reinforcement learning setting.
"""

import os
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class fcnn(nn.Module):
    """Fully-Connected Neural network for NEURWIN to modify its parameters"""

    def __init__(self, stateSize):
        super(fcnn, self).__init__()
        self.linear1 = nn.Linear(stateSize, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def printNumParams(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')
        print(f'Total number of trainable parameters: {total_params_trainable}')


class NEURWIN(object):

    def __init__(self, stateSize, lr, env, seed, sigmoidParam, numEpisodes, noiseVar,
                 batchSize, discountFactor, saveDir, episodeSaveInterval):
        # -------------constants-------------
        self.seed = seed
        torch.manual_seed(self.seed)
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed)  # create a special PRNG for a class instantiation

        self.numEpisodes = numEpisodes
        self.episodeRanges = np.arange(0, self.numEpisodes + episodeSaveInterval,
                                       episodeSaveInterval)
        self.stateSize = stateSize
        self.batchSize = batchSize
        self.sigmoidParam = sigmoidParam
        self.initialSigmoidParam = sigmoidParam
        self.beta = discountFactor
        self.env = env
        self.nn = fcnn(self.stateSize)
        self.linear1WeightGrad = []
        self.linear2WeightGrad = []
        self.linear3WeightGrad = []

        self.linear1BiasGrad = []
        self.linear2BiasGrad = []
        self.linear3BiasGrad = []

        self.paramChange = []
        self.numOfActions = 2
        self.directory = saveDir
        self.noiseVar = noiseVar

        self.temp = None
        self.LearningRate = lr
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.LearningRate)
        # -------------counters-------------
        self.currentMiniBatch = 0
        self.batchCounter = 0
        self.episodeCosts = []
        self.discountCosts = []
        # self.continueLearning()

        self.end = None
        self.episodeTimeStep = None
        self.episodeTimeList = None
        self.totalTimestep = None
        self.cost = None
        self.currentEpisode = None
        self.start = None

    def continueLearning(self):
        """Function for continuing with a learned model. Type in the number of episodes to
        continue from in trainedNumEpisodes """
        self.nn.load_state_dict(
            torch.load(
                self.directory + f'seed_{self.seed}_lr_{self.LearningRate}_batchSize_{5}'
                                 f'_trainedNumEpisodes_{100000}/trained_model.pt'
            )
        )

    def changeSigmoidParam(self):
        """Function for changing the sigmoid value as training happens. If not active,
        then m value is a constant. """
        self.sigmoidParam = self.sigmoidParam - self.sigmoidParam * 0.01
        if self.sigmoidParam <= 0.000001:
            self.sigmoidParam = 0.000001

    def newMiniBatchReset(self):
        """Function for new mini-batch procedures. For recovering bandits, the actviation cost is
        chosen for a random state. """

        # To run the code for restart environment
        state_arm = np.random.choice(self.env.num_states)
        state_thr = np.random.uniform(0, self.env.threshold)
        stateVal = np.array([state_arm, state_thr], dtype=np.float32)
        self.cost = self.nn.forward(stateVal).detach().numpy()[0]

    def takeAction(self, state):
        """Function for taking action based on the sigmoid function's generated probability
        distribution. """

        index = self.nn.forward(state)
        if (self.env.episodeTime == 0) and (self.currentEpisode % self.batchSize == 0):
            print(f'new state: {state}')
            self.newMiniBatchReset()

        sigmoidProb = torch.sigmoid(self.sigmoidParam * (index - self.cost))
        probOne = sigmoidProb.detach().numpy()[0]
        probs = [probOne, 1 - probOne]
        probs = np.array(probs)
        probs /= probs.sum()

        action = self.G.choice([1, 0], 1, p=probs)
        if action == 1:
            logProb = torch.log(sigmoidProb)
            logProb.backward()
        elif action == 0:
            logProb = torch.log(1 - sigmoidProb)
            logProb.backward()

        return action[0]

    def _saveEpisodeGradients(self):
        """Function for saving the gradients of each episode in one mini-batch"""

        self.linear1WeightGrad.append(self.nn.linear1.weight.grad.clone())
        self.linear2WeightGrad.append(self.nn.linear2.weight.grad.clone())
        self.linear3WeightGrad.append(self.nn.linear3.weight.grad.clone())

        self.linear1BiasGrad.append(self.nn.linear1.bias.grad.clone())
        self.linear2BiasGrad.append(self.nn.linear2.bias.grad.clone())
        self.linear3BiasGrad.append(self.nn.linear3.bias.grad.clone())

        self.optimizer.zero_grad()

    def _performBatchStep(self):
        """Function for performing the gradient ascent step on accumelated mini-batch gradients."""
        print('performing batch gradient step')

        meanBatchReward = sum(self.discountCosts) / len(self.discountCosts)
        for i in range(len(self.discountCosts)):
            self.discountCosts[i] = self.discountCosts[i] - meanBatchReward

            if self.nn.linear1.weight.grad is None:
                self.nn.linear1.weight.grad = self.discountCosts[i] * self.linear1WeightGrad[i]
                self.nn.linear1.bias.grad = self.discountCosts[i] * self.linear1BiasGrad[i]
            else:
                self.nn.linear1.weight.grad -= self.discountCosts[i] * self.linear1WeightGrad[i]
                self.nn.linear1.bias.grad -= self.discountCosts[i] * self.linear1BiasGrad[i]

            if self.nn.linear2.weight.grad is None:
                self.nn.linear2.weight.grad = self.discountCosts[i] * self.linear2WeightGrad[i]
                self.nn.linear2.bias.grad = self.discountCosts[i] * self.linear2BiasGrad[i]
            else:
                self.nn.linear2.weight.grad -= self.discountCosts[i] * self.linear2WeightGrad[i]
                self.nn.linear2.bias.grad -= self.discountCosts[i] * self.linear2BiasGrad[i]

            if self.nn.linear3.weight.grad is None:
                self.nn.linear3.weight.grad = self.discountCosts[i] * self.linear3WeightGrad[i]
                self.nn.linear3.bias.grad = self.discountCosts[i] * self.linear3BiasGrad[i]
            else:
                self.nn.linear3.weight.grad -= self.discountCosts[i] * self.linear3WeightGrad[i]
                self.nn.linear3.bias.grad -= self.discountCosts[i] * self.linear3BiasGrad[i]

        self.linear1WeightGrad = []
        self.linear2WeightGrad = []
        self.linear3WeightGrad = []

        self.linear1BiasGrad = []
        self.linear2BiasGrad = []
        self.linear3BiasGrad = []

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.discountCosts = []

        # self.changeSigmoidParam() # uncomment this to change m value every mini-batch

    def _discountCosts(self, costs):
        """Function for discounting an episode's reward based on set discount factor."""
        for i in range(len(costs)):
            costs[i] = (self.beta ** i) * costs[i]
        return sum(costs)

    def learn(self):
        self.start = time.time()
        self.currentEpisode = 0
        self.totalTimestep = 0
        self.episodeTimeStep = 0
        self.episodeTimeList = []
        # self.currentEpisode = 100 # for continuing learning

        while self.currentEpisode < self.numEpisodes:
            if self.currentEpisode in self.episodeRanges:
                self.close(self.currentEpisode)
            episodeCosts = []
            s_0 = self.env.reset()

            done = False
            # uncomment this for doing param change every timestep in episode
            # self.sigmoidParam = self.initialSigmoidParam

            while not done:
                action = self.takeAction(s_0)
                s_1, cost, done, info = self.env.step(action)

                if action == 1:
                    cost += self.cost
                episodeCosts.append(cost)
                s_0 = s_1
                # uncomment this for doing param change every timestep in episode
                # self.changeSigmoidParam()
                self.totalTimestep += 1
                self.episodeTimeStep += 1
                if done:
                    print(f'finished episode: {self.currentEpisode + 1}')
                    self.discountCosts.append(self._discountCosts(episodeCosts))
                    self.batchCounter += 1

                    self.episodeCosts.append(sum(episodeCosts))
                    self._saveEpisodeGradients()
                    episodeCosts = []
                    self.currentEpisode += 1
                    self.episodeTimeList.append(self.episodeTimeStep)
                    self.episodeTimeStep = 0
                    # uncomment this to change param every episode in one mini-batch
                    # self.changeSigmoidParam()

                    if self.batchCounter == self.batchSize:
                        self._performBatchStep()
                        self.currentMiniBatch += 1
                        self.batchCounter = 0
                        # uncomment this to change m value every episode in one mini-batch
                        # self.sigmoidParam = self.initialSigmoidParam
        self.end = time.time()
        self.close(self.numEpisodes)
        self.trainingEnding()
        print(f'---------------------------\nDONE. Time taken: {self.end - self.start:.5f} seconds.')
        print(f'total timesteps taken: {self.totalTimestep}')

    def close(self, episode):
        """Function for saving the NN parameters at defined interval *episodeSaveInterval* """

        directory = (f'{self.directory}' + f'seed_{self.seed}_lr_{self.LearningRate}_batchSize_'
                                           f'{self.batchSize}_trainedNumEpisodes_{episode}')
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.nn.state_dict(), directory + '/trained_model.pt')

    def trainingEnding(self):
        """Function for saving training information once it is over."""

        file = open(self.directory + 'trainingInfo.txt', 'w+')
        file.write(f'training time: {self.end - self.start:.5f} seconds\n')
        file.write(f'training episodes: {self.numEpisodes}\n')
        file.write(f'Mini-batch size: {self.batchSize}\n')
        file.write(f'Total timesteps: {self.totalTimestep}\n')
        file.close()

        data = {'episode': range(len(self.episodeTimeList)),
                'episode_timesteps': self.episodeTimeList}
        df = pd.DataFrame(data=data)
        df.to_csv(self.directory + f'episode_timesteps_batchsize_{self.batchSize}.csv', index=False)
