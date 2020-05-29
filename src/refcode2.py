# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.midWidth = gameState.data.layout.width/2
    self.midHeight = gameState.data.layout.height/2
    CaptureAgent.registerInitialState(self, gameState)
    self.nonWallPos = gameState.data.layout.walls.asList(False)
    self.posBelief = {}
    self.eatenFood = util.Counter()
    self.nTime = 0
    self.enemyPos = {}
    self.enemyDist = []
    self.firstTarget = 1
    self.seenByEnemy = util.Counter()
    self.strategy = 'Attack'
    
    for enemy in self.getOpponents(gameState):
      self.initializeBelief(enemy)
    self.foodRisk = util.Counter()
    for p in self.nonWallPos:
      self.foodRisk[p] = p
    self.safePos = []
    if self.red:
      self.unsafeList = [(21,1), (23,1), (25,1), (26,1), (28,1), (23,3), (26,3), (28,3), (21,4), (22,4), (23,4), (28, 4),
      (28,5), (17,6), (20,6), (21,6), (23,6), (24,6), (25,6), (24,7), (28,7), (28,9), (22,10), (28,10), (28,11), (21,12), (23,12),
      (24,12), (28,12), (28, 13), (26,14), (28,14)]
    else:
      self.unsafeList = [(31-21,15-1), (31-23,15-1), (31-25,15-1), (31-26,15-1), (31-28,15-1), (31-23,15-3), (31-26,15-3), (31-28,15-3), (31-21,15-4), (31-22,15-4), (31-23,15-4), (31-28, 15-4),
      (31-28,15-5), (31-17,15-6), (31-21,15-6), (31-23,15-6), (31-24,15-6), (31-25,15-6), (31-24,15-7), (31-28,15-7), (31-28,15-9), (31-22,15-10), (31-28,15-10), (31-28,15-11), (31-21,15-12), (31-23,15-12),
      (31-24,15-12), (31-28,15-12), (31-28, 15-13), (31-26,15-14), (31-28,15-14)]
    self.indexList = [(21,1), (23,1), (25,1), (26,1), (26,3), (21,4), (22,4), (23,4), (23,3), (17,6), (20,6), (21,6),
    (23,6), (24,6), (24,7), (25,6), (28,7), (21,8), (22,10), (21,12), (23,12), (24,12), (26,14), (28,9), (28,10), (28,11),
    (28,12), (28,13), (28,14)]
    self.itemList = [(21,2), (23,2), (25,2), (25,2), (25,3), (23,2), (23,2), (23,2), (23,2), (17,7), (19,6), (19,6),
    (25,5), (25,5), (25,5), (25,5), (27,7), (21,9), (21,9), (20,12), (24,13), (24,13), (26,13), (27,9), (27,9), (27,9),
    (27,9), (27,9), (27,9)]
    if self.red:
      for n in range(0, len(self.indexList) - 1):
        self.foodRisk[self.indexList[n]] = self.itemList[n]
    else:
      for n in range(0, len(self.indexList) - 1):
        self.foodRisk[(31 - self.indexList[n][0], 15 - self.indexList[n][1])] = (31 - self.itemList[n][0], 15 - self.itemList[n][1])
    if self.index == 0:
      self.initialTarget = (15, 14)
    elif self.index == 1:
      self.initialTarget = (16, 1)
    elif self.index == 2:
      self.initialTarget = (15, 1)
    else:
      self.initialTarget = (16, 14)
    self.target = self.initialTarget

  def followTarget(self, gameState, agent):
    myPos = gameState.getAgentPosition(agent)
    posX = myPos[0]
    posY = myPos[1]
    bestAction = None
    bestDist = None
    for p in [(posX - 1, posY), (posX + 1, posY), (posX, posY - 1), (posX, posY + 1)]:
      if p in self.nonWallPos:
        currDist = self.getMazeDistance(p, self.target)
        if bestDist == None or currDist < bestDist:
          bestDist = currDist
          bestAction = p
    return bestAction

  def canEatFood(self, gameState, agent, target):
    myPos = gameState.getAgentPosition(agent)
    teamIndex = (self.index + 2) % 4
    dest = self.foodRisk[target]
    risk = util.manhattanDistance(dest, target) * 2
    myDist = self.getMazeDistance(myPos, dest)
    enemyIndex = (self.index + 1) % 4
    if target == (28,13) or target == (28,14) or target == (3,2) or target == (3,1):
      return False
    elif self.enemyScaredTime(enemyIndex, gameState) == 0:
      if any([self.seenByEnemy[(agent, opponent)] == 1 and self.seenByEnemy[(teamIndex, opponent)] == 0 for opponent in self.getOpponents(gameState)]):
        if self.foodRisk[target] == self.foodRisk[self.myPos]:
          if any([self.getMazeDistance(dest, self.enemyPos[opponent]) <= util.manhattanDistance(myPos, target) * 2 + util.manhattanDistance(myPos, dest) + 1 for opponent in self.getOpponents(gameState)]):
            return False
          else:
            return True
        else:
          if any([self.getMazeDistance(dest, self.enemyPos[opponent]) <= myDist + risk + 1 for opponent in self.getOpponents(gameState)]):
            return False
          else:
            return True
      else:
        return True
    else:
      if myDist + risk <= self.enemyScaredTime(enemyIndex, gameState):
        return True
      else:
        return False

  def getAvailableFoodList(self, gameState, agent):
    self.myPos = gameState.getAgentPosition(agent)
    foodList = self.getFood(gameState).asList()
    availableFoodList = []
    for food in foodList:
      if self.canEatFood(gameState, agent, food):
        availableFoodList.append(food)
    #print(availableFoodList)
    return availableFoodList

  def renewTarget(self, target):
    self.target = target

  def getClosestFood(self, gameState, agent, foodList):
    self.myPos = gameState.getAgentPosition(self.index)
    closestFoodDist = None
    closestFoodPos = None
    if foodList == []:
      return self.initialTarget
    else:
      for food in foodList:
        currFoodDist = self.getMazeDistance(food, self.myPos)
        if closestFoodPos is None or currFoodDist < closestFoodDist:
          closestFoodPos = food
          closestFoodDist = currFoodDist
      return closestFoodPos


  def initializeBelief(self, agent):
    self.posBelief[agent] = util.Counter()
    for p in self.nonWallPos:
      self.posBelief[agent][p] = 0.5

  def elapseTime(self, enemy, gameState):
    newBelief = util.Counter()
    for (oldX, oldY), oldProb in self.posBelief[enemy].items():
      newProb = util.Counter()
      for p in [(oldX - 1, oldY), (oldX + 1, oldY), (oldX, oldY - 1), (oldX, oldY + 1)]:
        if p in self.nonWallPos:
          newProb[p] = 1.0
      newProb.normalize()
      for newPos, prob in newProb.items():
        newBelief[newPos] += prob * oldProb
    newBelief.normalize()
    self.posBelief[enemy] = newBelief
    thiefIndex = (self.index - 1) % 4
    lostFood = []
    lastObserved = self.getPreviousObservation()
    if lastObserved:
      lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList() if food not in self.getFoodYouAreDefending(gameState).asList()]
    if len(lostFood) != 0:
      f = lostFood[0]
      newBelief = util.Counter()
      if self.eatenFood[f] != 1:
        self.eatenFood[f] = 1
        newBelief[f] = 1.0
        self.posBelief[thiefIndex] = newBelief

  def guessPosition(self, agent, gameState):
    result = self.posBelief[agent].argMax()
    if self.posBelief[agent][result] == 1:
      return result
    else:
      return gameState.getInitialAgentPosition(agent)

  def observe(self, opponent, observation, gameState):
    myPos = gameState.getAgentPosition(self.index)
    teamPos = [gameState.getAgentPosition(team) for team in self.getTeam(gameState)]
    newBelief = util.Counter()
    noisyDist = observation[opponent]
    for p in self.nonWallPos:
      if self.red:
        isPac = p[0] < self.midWidth
      else:
        isPac = p[0] > self.midWidth
      if any([util.manhattanDistance(teamPosition, p) <= 5 for teamPosition in teamPos]):
        newBelief[p] = 0.0
      elif isPac != gameState.getAgentState(opponent).isPacman:
        newBelief[p] = 0.0
      else:
        trueDist = util.manhattanDistance(myPos, p)
        posProb = gameState.getDistanceProb(trueDist, noisyDist)
        newBelief[p] = posProb * self.posBelief[opponent][p]
    if newBelief.totalCount() == 0:
      self.initializeBelief(opponent)
    else:
      newBelief.normalize()
      self.posBelief[opponent] = newBelief

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    teamIndex = (self.index + 2) % 4
    self.myPos = gameState.getAgentPosition(self.index)
    enemyIndex = (self.index - 1) % 4
    if gameState.getAgentPosition(self.index) == self.target:
      self.firstTarget = 0
    noisyDistances = gameState.getAgentDistances()
    for opponent in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opponent)
      if pos:
        if util.manhattanDistance(self.myPos, pos) < 6:
          self.seenByEnemy[(self.index, opponent)] = 1
        else:
          self.seenByEnemy[(self.index, opponent)] = 0
        newBelief = util.Counter()
        newBelief[pos] = 1.0
        self.posBelief[opponent] = newBelief
      else:
        self.seenByEnemy[(self.index, opponent)] = 0
        self.elapseTime(opponent, gameState)
        self.observe(opponent, noisyDistances, gameState)
    #for opponent in self.getOpponents(gameState):
      #print('agent', opponent)
      #for pos, prob in self.posBelief[opponent].items():
        #if prob > 0:
          #print(pos, prob)
    self.enemyDist = []
    for opponent in self.getOpponents(gameState):
      self.enemyPos[opponent] = self.guessPosition(opponent, gameState)
      self.enemyDist.append(self.getMazeDistance(self.myPos, self.enemyPos[opponent]))
    print('Enemy Position: ', self.enemyPos)
    print('My Position: ', self.myPos)
    print('Seen by Enemy: ', self.seenByEnemy)
    self.availableFoodList = self.getAvailableFoodList(gameState, self.index)
    #Experiment1
    if len(self.availableFoodList) == 0 and self.agentIsPacman(self.index, gameState) == 1:
      self.backHome = 1
    #print(self.firstTarget)
    if self.firstTarget == 0:
      self.target = self.getClosestFood(gameState, self.index, self.availableFoodList)
    #print(self.getClosestFood(gameState, self.index, self.availableFoodList))
    self.displayDistributionsOverPositions(self.posBelief.values())
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    print('agent', self.index)
    print(values)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def agentIsPacman(self, agent, gameState):
    agentPos = gameState.getAgentPosition(agent)
    return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

  def enemyScaredTime(self, enemy, gameState):
    enemyState = gameState.getAgentState(self.index)
    return enemyState.scaredTimer

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def chooseAction(self, gameState):
    enemyIndex = (self.index + 1) % 4
    if (gameState.data.timeleft + 40)/4 <= 50 and self.agentIsPacman(self.index, gameState) == 1:
      self.backHome = True
    elif (gameState.getAgentState(self.index).numCarrying < 5 and len(self.getFood(gameState).asList())):
      self.backHome = False
    elif len(self.getFood(gameState).asList()) < 3:
      self.backHome = True
    else:
      self.backHome = True
    return ReflexCaptureAgent.chooseAction(self, gameState)

  def getFeatures(self, gameState, action):
    if self.strategy == 'Attack':
      if self.backHome:
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        prevPosition = gameState.getAgentPosition(self.index)
        myPosition = successor.getAgentPosition(self.index)
        startPos = gameState.getInitialAgentPosition(self.index)
        distanceFromStart = self.getMazeDistance(myPosition, startPos)
        features['distanceFromStart'] = distanceFromStart
        dists = [self.getMazeDistance(myPosition, self.enemyPos[enemy]) for enemy in self.getOpponents(gameState)]
        distanceFromEnemy = min(dists)
        print(myPosition[0] - prevPosition[0], myPosition[1] - prevPosition[1])
        print('Distance from enemy: ', distanceFromEnemy)
        features['distanceFromEnemy<2'] = distanceFromEnemy < 2
        if myPosition in self.unsafeList:
          inUnsafePos = 1
        else:
          inUnsafePos = 0
        features['unsafePos'] = inUnsafePos
        if myPosition == gameState.getInitialAgentPosition(self.index):
          getEaten = 1
        else:
          getEaten = 0
        features['getEaten'] = getEaten
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
          features['reverse'] = 1
        features['stayStill'] = (myPosition == prevPosition)
        if any([(myPosition == enemyPosition) for enemyPosition in self.enemyPos]):
          features['eatEnemy'] = 1
        else:
          features['eatEnemy'] = 0
        print(features)
        return features
      else:
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPosition = successor.getAgentPosition(self.index)
        prevPosition = gameState.getAgentPosition(self.index)
        teamIndex = (self.index + 2) % 4
        dists = [self.getMazeDistance(myPosition, self.enemyPos[enemy]) for enemy in self.getOpponents(gameState)]
        distanceFromEnemy = min(dists)
        print(myPosition[0] - prevPosition[0], myPosition[1] - prevPosition[1])
        print('Distance from enemy: ', distanceFromEnemy)
        features['distanceFromEnemy<2'] = (distanceFromEnemy > 0 and distanceFromEnemy < 2)
        if myPosition in self.unsafeList:
          inUnsafePos = 1
        else:
          inUnsafePos = 0
        features['unsafePos'] = inUnsafePos
        if myPosition == gameState.getInitialAgentPosition(self.index):
          getEaten = 1
        else:
          getEaten = 0
        features['getEaten'] = getEaten
        if successor.getAgentPosition(self.index) == self.followTarget(gameState, self.index):
          followedTarget = 1
        else:
          followedTarget = 0
        features['followTarget'] = followedTarget
        if any([(myPosition == enemyPosition) for enemyPosition in self.enemyPos]):
          features['eatEnemy'] = 1
        else:
          features['eatEnemy'] = 0
        print(features)
        return features
    elif self.strategy == 'Defense':
      if self.getAgentState(self.index).scaredTimer == 0:
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPosition = successor.getAgentPosition(self.index)
        prevPosition = gameState.getAgentPosition(self.index)
        teamIndex = (self.index + 2) % 4
        dists = [self.getMazeDistance(myPosition, self.enemyPos[enemy]) for enemy in self.getOpponents(gameState)]
        distanceFromEnemy = min(dists)
        print(myPosition[0] - prevPosition[0], myPosition[1] - prevPosition[1])
        print('Distance from enemy: ', distanceFromEnemy)
        features['distanceFromEnemy<2'] = (distanceFromEnemy > 0 and distanceFromEnemy < 2)
        if myPosition in self.unsafeList:
          inUnsafePos = 1
        else:
          inUnsafePos = 0
        features['unsafePos'] = inUnsafePos
        if myPosition == gameState.getInitialAgentPosition(self.index):
          getEaten = 1
        else:
          getEaten = 0
        features['getEaten'] = getEaten
        if successor.getAgentPosition(self.index) == self.followTarget(gameState, self.index):
          followedTarget = 1
        else:
          followedTarget = 0
        features['followTarget'] = followedTarget
        if any([(myPosition == enemyPosition) for enemyPosition in self.enemyPos]):
          features['eatEnemy'] = 1
        else:
          features['eatEnemy'] = 0
        print(features)
        return features
      #else:

      



  def getWeights(self, gameState, action):
    if self.strategy == 'Attack':
        if self.backHome:
          return {'distanceFromStart': -10, 'reverse': -2, 'distanceFromEnemy<2': -100, 'unsafePos': -50,
          'getEaten': -1000}
        else:
          return {'followTarget': 50, 'distanceFromEnemy<2': -100, 'unsafePos': -20, 'getEaten': -1000}
    #elif self.strategy == 'Defense':


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
