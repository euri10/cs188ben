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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        originalscore = currentGameState.getScore()
        distscore = 0
        getFscore = 0
        ghostScore = 0
        stopScore = 0
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            getFscore = 100
        else:
            getFscore = 0
        fdist = [util.manhattanDistance(newPos, f) for f in newFood.asList()]
        if len(fdist) > 0:
            # print fdist
            distscore = 1.0 / min(fdist)

        gdist = [util.manhattanDistance(newPos, g.getPosition()) for g in newGhostStates]
        if len(gdist) > 0:
            # print gdist
            if max(gdist) == 0.0:
                ghostScore = -100
            else:
                ghostScore = -1.0 / max(gdist)

        if action == 'Stop':
            stopScore = -10

        score = originalscore + distscore + getFscore + ghostScore + stopScore
        # print('{}|{}, orig: {}, dist: {}, getF: {}, ghost: {}, stop: {}'.format(action, score, originalscore, distscore, getFscore, ghostScore, stopScore))
        return score


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

        def tourMax(gameState, depth):
            depth -= 1
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = float('-inf')
            for ac in gameState.getLegalActions(0):
                u = max(u, tourMin(gameState.generateSuccessor(0, ac), depth, 1))
            return u

        def tourMin(gameState, depth, agent):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = float('inf')
            for ac in gameState.getLegalActions(agent):
                if agent == ghosts:
                    u = min(u, tourMax(gameState.generateSuccessor(agent, ac), depth))
                else:
                    u = min(u, tourMin(gameState.generateSuccessor(agent, ac), depth, agent + 1))
            return u

        ghosts = gameState.getNumAgents() - 1
        action = Directions.STOP
        score = float('-inf')
        for ac in gameState.getLegalActions():
            prevscore = score
            score = max(score, tourMin(gameState.generateSuccessor(0, ac), self.depth, 1))
            if score > prevscore:
                action = ac
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def tourMax(gameState, alpha, beta, depth):
            depth -= 1
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = float('-inf')
            for ac in gameState.getLegalActions(0):
                u = max(u, tourMin(gameState.generateSuccessor(0, ac), alpha, beta, depth, 1))
                if u > beta:
                    return u
                alpha = max(alpha, u)
            return u

        def tourMin(gameState, alpha, beta, depth, agent):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = float('inf')
            for ac in gameState.getLegalActions(agent):
                if agent == ghosts:
                    u = min(u, tourMax(gameState.generateSuccessor(agent, ac), alpha, beta, depth))
                else:
                    u = min(u, tourMin(gameState.generateSuccessor(agent, ac), alpha, beta, depth, agent + 1))
                if u < alpha:
                    return u
                beta = min(beta, u)
            return u

        ghosts = gameState.getNumAgents() - 1
        action = Directions.STOP
        score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for ac in gameState.getLegalActions():
            nextS = gameState.generateSuccessor(0, ac)
            prevscore = score
            score = max(score, tourMin(nextS, alpha, beta, self.depth, 1))
            if score > beta:
                return score
            alpha = max(alpha, score)
            if score > prevscore:
                action = ac
        return action


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
        def tourMax(gameState, depth):
            depth -= 1
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = float('-inf')
            for ac in gameState.getLegalActions(0):
                u = max(u, expexted(gameState.generateSuccessor(0, ac), depth, 1))
            return u

        def expexted(gameState, depth, agent):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            u = 0.0
            for ac in gameState.getLegalActions(agent):
                if agent == ghosts:
                    u += (tourMax(gameState.generateSuccessor(agent, ac), depth))/len(gameState.getLegalActions(agent))
                else:
                    u += (expexted(gameState.generateSuccessor(agent, ac), depth, agent + 1))/len(gameState.getLegalActions(agent))
            return u

        ghosts = gameState.getNumAgents() - 1
        action = Directions.STOP
        score = float('-inf')
        for ac in gameState.getLegalActions():
            prevscore = score
            score = max(score, expexted(gameState.generateSuccessor(0, ac), self.depth, 1))
            if score > prevscore:
                action = ac
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    # score better if ghosts reachable once pellet eaten. calculate the number of reachable ghosts
    # ghTimers is the list of (timers on ghost, distance to ghost) for scared ghost
    ghTimers = [(s.scaredTimer, util.manhattanDistance(currentGameState.getPacmanPosition(), s.getPosition())) for s in currentGameState.getGhostStates() if s.scaredTimer>0]
    # print '(timer, distance)', ghTimers
    # ghIsFood is the list of scared ghost whose distance to reach is inferior than their timer, hence they are reachable
    ghIsFood = [g[1] for g in ghTimers if g[0] > g[1]]
    # print 'food', ghIsFood
    if len(ghIsFood) > 0:
        score += 10.0/min(ghIsFood)
    return score

# Abbreviation
better = betterEvaluationFunction

