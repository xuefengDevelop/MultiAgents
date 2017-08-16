# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(0,self.iterations):
            value =  util.Counter()
            for state in self.mdp.getStates():
                bestActionValue = - float('inf')                        # declare best score and best move to record
                bestMove = None
                if self.mdp.isTerminal(state):                          # check if it is terminal states
                    value[state] = 0
                    continue
                for action in mdp.getPossibleActions(state):                # all action in mpd choose best action to get highest points
                    actionPionts = self.getQValue(state,action)
                    if bestActionValue < actionPionts:
                        bestActionValue =actionPionts
                        bestMove = action
                value[state] = bestActionValue
            self.values = value                                         # set the value to value list of best value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        for trans,prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, trans)
            total += prob * (reward + (self.discount * self.values[trans]))  # Q value compute function
    
        return total
    

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestValue = - float('inf')
        bestMove = None
    
        if self.mdp.isTerminal(state):
            return None

        for action in self.mdp.getPossibleActions(state):
            actionPoints = self.getQValue(state, action)
            if bestValue < actionPoints:
                bestMove = action
                bestValue = actionPoints

        return bestMove


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
