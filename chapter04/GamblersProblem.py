#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
# goal
GOAL = 100

# all states, including state 0 and state 100
states = np.arange(GOAL + 1)

# probability of head
headProb = 0.4

# optimal policy
policy = np.zeros(GOAL + 1)

# state value
stateValue = np.zeros(GOAL + 1)
stateValue[GOAL] = 1.0

# Number of sweeps through the state set
sweepCount = 0

#small positive number
theta = 1e-25

# value iteration
while True:
    delta = 0.0
    sweepCount +=1
    for state in states[1:GOAL]:
        # get possilbe actions for current state
        actions = np.arange(1, min(state, GOAL - state) + 1)
        actionReturns = []
        for action in actions:
            actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
        newValue = np.max(actionReturns)
        delta = max(np.abs(stateValue[state] - newValue), delta)
        # update state value
        stateValue[state] = newValue
    if delta < theta:
        print("Total number of sweeps through state set: " + str(sweepCount))
        break

# calculate the optimal policy
for state in states[1:GOAL]:
    # Actions start from $1 stakes
    actions = np.arange(1, min(state, GOAL - state) + 1)
    actionReturns = []
    for action in actions:
        actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
    # due to tie, can't reproduce the optimal policy in book
    policy[state] = actions[np.argmax(actionReturns)]

# figure 4.3
plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
#Plotting with a step function is better to discern between discrete values
plt.step(states, stateValue)
# Set 5 ticks on the x-axis, so as to get an interval of 25 units between each other
plt.gca().locator_params(nbins=5, axis='x')
plt.figure(2)
plt.step(states, policy)
plt.gca().locator_params(nbins=5, axis='x')
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()
