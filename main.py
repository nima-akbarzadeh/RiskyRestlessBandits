import os
import sys
import gym
import time
import torch
import random
import numpy as np
from neurwin_risky import NEURWIN

sys.path.insert(0, 'envs/')
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
from envs.RestartBanditEnv import RestartBanditEnv

# from deadlineSchedulingMultipleArmsEnv import deadlineSchedulingMultipleArmsEnv

# from sizeAwareIndexEnv import sizeAwareIndexEnv
# from recoveringBanditsEnv import recoveringBanditsEnv
# from sizeAwareIndexMultipleArmsEnv import sizeAwareIndexMultipleArmsEnv
# from recoveringBanditsMultipleArmsEnv import recoveringBanditsMultipleArmsEnv


########################### PARAMETERS ########################################
STATESIZE = 2
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# sigmoid function sensitivity parameter. 1 for deadline. 5 for recovering. 0.75 for size-aware
sigmoidParam = 5
# gradient ascent step every n episodes
BATCHSIZE = 5
learningRate = 1e-03
numEpisodes = 30000
discountFactor = 0.99

TRAIN = True
# timesteps for training
EPISODELIMIT = 300
noiseVar = 0.0
#####################################################
# for size-aware index's cases
CASE = 1
CLASSVAL = 1

if CASE == 1 and CLASSVAL == 1:
    HOLDINGCOST = 1
    GOODTRANS = 33600
    BADTRANS = 8400
    GOODPROB = 0.75
    LOAD = 1000000

elif CASE == 1 and CLASSVAL == 2:
    HOLDINGCOST = 1
    GOODTRANS = 33600
    BADTRANS = 8400
    GOODPROB = 0.1
    LOAD = 1000000

else:
    print(f'entered case not in list. Exiting...')
    exit(1)

# #######################################----TRAINING
# SETTINGS----#########################################################
# ######################################### NEURWIN DEADLINE SCHEDULING
# ###################################################

if __name__ == '__main__':
    ################################## RESTART BANDIT #########################################
    if noiseVar > 0:
        restartDirectory = f'trainResults/neurwin/restart_env/noise_{noiseVar}_version/'
    else:
        restartDirectory = f'trainResults/neurwin/restart_env/'

    NUMSTATES = 3
    FTYPE = 'linear'
    RESETC = 1.5
    PROBR = 0.5 * np.ones(1)
    TTYPE = 1
    DISCOUNT = 0.99
    EPISODELIMIT = 100
    CTHRESHOLD = 0.9
    restartEnv = RestartBanditEnv(seed=SEED, numEpisodes=numEpisodes, episodeLimit=EPISODELIMIT,
                                  num_states=NUMSTATES, function_type=FTYPE, reset_coef=RESETC,
                                  prob_remain=PROBR, transition_type=TTYPE, discount=DISCOUNT,
                                  cost_threshold=CTHRESHOLD,
                                  train=TRAIN, batchSize=BATCHSIZE, noiseVar=noiseVar)

    agent = NEURWIN(stateSize=STATESIZE, lr=learningRate, env=restartEnv, sigmoidParam=sigmoidParam,
                    numEpisodes=numEpisodes, noiseVar=noiseVar, seed=SEED, batchSize=BATCHSIZE,
                    discountFactor=DISCOUNT, saveDir=restartDirectory,
                    episodeSaveInterval=5)
    agent.learn()

    ################################## DEADLINE SCHEDULING #########################################
    # if noiseVar > 0:
    #     deadlineDirectory = f'trainResults/neurwin/deadline_env/noise_{noiseVar}_version/'
    # else:
    #     deadlineDirectory = f'trainResults/neurwin/deadline_env/'
    #
    # deadlineEnv = deadlineSchedulingEnv(seed=SEED, numEpisodes=numEpisodes,
    #                                     episodeLimit=EPISODELIMIT, maxDeadline=12, maxLoad=9,
    #                                     newJobProb=0.7, processingCost=0.5, train=TRAIN,
    #                                     batchSize=BATCHSIZE, noiseVar=noiseVar)
    #
    # agent = NEURWIN(stateSize=STATESIZE, lr=learningRate, env=deadlineEnv, sigmoidParam=sigmoidParam,
    #                 numEpisodes=numEpisodes, noiseVar=noiseVar, seed=SEED, batchSize=BATCHSIZE,
    #                 discountFactor=discountFactor, saveDir=deadlineDirectory,
    #                 episodeSaveInterval=5)
    # agent.learn()

    ################################## NEURWIN WIRELESS SCHEDULING ##################################
    '''
    if noiseVar > 0:
        sizeAwareDirectory = (f'trainResults/neurwin/size_aware_env/noise_{noiseVar}_version/case_{CASE}/class_{CLASSVAL}/')
    else:
        sizeAwareDirectory = (f'trainResults/neurwin/size_aware_env/case_{CASE}/class_{CLASSVAL}/')
    
    sizeAwareEnv = sizeAwareIndexEnv(numEpisodes=numEpisodes, HOLDINGCOST=HOLDINGCOST, seed=SEED, Training=TRAIN, r1=BADTRANS,
    r2=GOODTRANS, q=GOODPROB, case=CASE, classVal=CLASSVAL, noiseVar = noiseVar, 
    load=LOAD, batchSize = BATCHSIZE, maxLoad = LOAD, episodeLimit=EPISODELIMIT, fixedSizeMDP=False)
    
    agent = NEURWIN(stateSize=STATESIZE,lr=learningRate, env=sizeAwareEnv, sigmoidParam=sigmoidParam, numEpisodes=numEpisodes, noiseVar=noiseVar,
    seed=SEED, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = sizeAwareDirectory, episodeSaveInterval=100)
    agent.learn()
    '''
    ################################## RECOVERING BANDITS ##################################
    '''
    maxWait = 20 # maximum time before refreshing the arm
    STATESIZE = 1
    CASE = 'A'  # A,B,C,D different recovery functions
    
    if CASE == 'A':
        THETA = [10., 0.2, 0.0]
    elif CASE == 'B':
        THETA = [8.5, 0.4, 0.0]
    elif CASE == 'C':
        THETA = [7., 0.6, 0.0]
    elif CASE == 'D':
        THETA = [5.5, 0.8, 0.0]
    
    
    if noiseVar > 0:
        recoveringDirectory = (f'trainResults/neurwin/recovering_bandits_env/noise_{noiseVar}_version/recovery_function_{CASE}/')
    else:
        recoveringDirectory = (f'trainResults/neurwin/recovering_bandits_env/recovery_function_{CASE}/')
    
    os.makedirs(recoveringDirectory)
    file = open(recoveringDirectory+'used_parameters.txt', 'w+')
    file.write(f'Theta0, Theta1, Theta2: {THETA}\n')
    file.write(f'max wait for recovery function: {maxWait}\n')
    file.close()
    
    print(f'selected theta: {THETA}')
    recoveringEnv = recoveringBanditsEnv(seed=SEED, numEpisodes=numEpisodes, episodeLimit=EPISODELIMIT, train=TRAIN, 
    batchSize=BATCHSIZE,thetaVals=THETA, noiseVar=noiseVar, maxWait = maxWait)
    
    agent = NEURWIN(stateSize=STATESIZE,lr=learningRate, env=recoveringEnv, noiseVar=noiseVar,
    sigmoidParam=sigmoidParam, numEpisodes=numEpisodes,seed=SEED, batchSize=BATCHSIZE, 
    discountFactor=discountFactor, saveDir = recoveringDirectory,episodeSaveInterval=100)
    agent.learn()
    '''
