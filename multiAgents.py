# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_distances = []
        ghost_distances = []

        for food in newFood.asList():
            food_distances.append(manhattanDistance(food, newPos))

        for ghost in successorGameState.getGhostPositions():
            ghost_distances.append(manhattanDistance(ghost, newPos))

        for distance in ghost_distances:
            if distance <= 1:  # the ghost is too close
                return -100000
        if len(food_distances) == 0:  # no food
            return 100000

        return 1 / sum(food_distances) + 1 / len(food_distances)
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agent, depth, gameState):
            scores = []
            # Base cases
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # maximizer
                for legalAction in gameState.getLegalActions(agent):
                    scores.append(minimax(1, depth, gameState.generateSuccessor(agent, legalAction)))
                return max(scores)

            else:  # minimizer
                nextAgent = agent + 1
                if gameState.getNumAgents() - 1 == agent:
                    nextAgent = 0
                    depth += 1
                for legalAction in gameState.getLegalActions(agent):
                    scores.append(minimax(nextAgent, depth, gameState.generateSuccessor(agent, legalAction)))
                return min(scores)

        maximum = - float("inf")  # setting the score to - infity so we can maximize later
        for legalAction in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, legalAction))
            if score > maximum:
                maximum = score
                action = legalAction

        return action
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maximizer(agent, depth, gameState, alpha, beta):  # maximizer function
            v = float("-inf")
            for newState in gameState.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, gameState.generateSuccessor(agent, newState), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minimizer(agent, depth, gameState, alpha, beta):  # minimizer function
            v = float("inf")
            next_agent = agent + 1
            if gameState.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1
            for newState in gameState.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, gameState.generateSuccessor(agent, newState), alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def alphabetaprune(agent, depth, gameState, alpha, beta):
            # Base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:  # maximizer
                return maximizer(agent, depth, gameState, alpha, beta)
            else:  # minimizer
                return minimizer(agent, depth, gameState, alpha, beta)

        score = - float("inf")
        alpha = - float("inf")
        beta = float("inf")
        for legalAction in gameState.getLegalActions(0):
            val = alphabetaprune(1, 0, gameState.generateSuccessor(0, legalAction), alpha, beta)
            if val > score:
                score = val
                action = legalAction
            if score > beta:
                return score
            alpha = max(alpha, score)

        return action
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agent, depth, gameState):
            scores = []
            # Base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # maximizing for pacman
                for legalAction in gameState.getLegalActions(agent):
                    scores.append(expectimax(1, depth, gameState.generateSuccessor(agent, legalAction)))
                return max(scores)
            else:  # we need to sum the utilities and then divide them by the number of legal actions
                nextAgent = agent + 1
                if gameState.getNumAgents() - 1 == agent:
                    nextAgent = 0
                    depth += 1
                for legalAction in gameState.getLegalActions(agent):
                    scores.append(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, legalAction)))

                return sum(scores) / float(len(gameState.getLegalActions(agent)))

        maximum = -float("inf")
        for legalAction in gameState.getLegalActions(0):
            score = expectimax(1, 0, gameState.generateSuccessor(0, legalAction))
            if score > maximum:
                maximum = score
                action = legalAction

        return action
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    I calculate the manhattan distance between Pacman and the closest food then remove it from the scores
    """
    "*** YOUR CODE HERE ***"
    foodList = []

    for food in currentGameState.getFood().asList():
        foodList.append(manhattanDistance(list(currentGameState.getPacmanPosition()), food))

    if len(foodList) == 0:
        foodList.append(0)

    return scoreEvaluationFunction(currentGameState) - min(foodList)
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


