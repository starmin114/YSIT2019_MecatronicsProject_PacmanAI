
# team5.py
# ----------------------------------------------------
# Basic Pacman AI code for IIT4312 Mechatronics Project
# by Team 5
# version 20190523

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import copy
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent1', second = 'Agent2'):
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
  # OUR BRAND NEW AGENT!!

  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    start = time.time()
    
    if (self.xSize == 0): # initially get map size
      map = gameState.data.layout.walls.data
      self.xSize = len(map)
      self.ySize = len(map[0])
      #print(self.xSize, self.ySize)

    self.sentryEnabled = (self.getScore(gameState) > 0 and not gameState.getAgentState(self.index).scaredTimer > 0)

    action = self.subchooseAction(gameState, 3)[1]
    self.foodMat = self.getFood(gameState.generateSuccessor(self.index, action)).asList()
    self.superfoodMat = self.getCapsules(gameState.generateSuccessor(self.index, action))

    print('%.4fs for %d' % (time.time() - start, self.index))
    return action

  def subchooseAction(self, gameState, n):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    bestaction = random.choice(actions)

    if n==1:
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestaction = random.choice(bestActions)
    else:
      maxValue = -9999
      for subaction in actions:
        successor = gameState.generateSuccessor(self.index, subaction)
        tempvalue = self.subchooseAction(successor, n-1)[0]
        if (tempvalue > maxValue and subaction != Directions.STOP): # don't choose STOP
          bestaction = subaction
          maxValue = self.evaluate(gameState, subaction)

    
    if n==3:
      # updating weights (1)
      gamma = 0.95
      successor = gameState.generateSuccessor(self.index, bestaction)

      lastGameState = self.getPreviousObservation()
      if (lastGameState != None):
        rewards = 10 * (self.getScore(gameState) - self.getScore(lastGameState))

        difference = rewards + gamma * maxValue - self.lastQ
        #print("Diff: {}, MaxQ: {}, LastQ: {}".format(difference, maxValue, self.lastQ))

        self.weights = self.updateWeights(lastGameState, self.lastaction, difference)
      self.lastQ = maxValue
      self.lastaction = bestaction
      '''
          foodLeft = len(self.getFood(gameState).asList())

          if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
              successor = self.getSuccessor(gameState, action)
              pos2 = successor.getAgentPosition(self.index)
              dist = self.getMazeDistance(self.start,pos2)
              if dist < bestDist:
                bestAction = action
                bestDist = dist
            return bestAction
      '''
      # print(self.getFeatures(gameState, action))
      # print("index:", self.index, self.weights)

    return (maxValue, bestaction)
  
  def getMinPalletDist(self, predecessor, gameState):
    foodmatrix = self.foodMat
    mindistance = 999
    selfpos = gameState.getAgentPosition(self.index)
    for i in range(len(foodmatrix)):
      distance = self.getMazeDistance((foodmatrix[i][0], foodmatrix[i][1]), selfpos)
      if distance < mindistance:
        mindistance = distance
    return mindistance

  def getourzone(self, gameState, red):
    if red:
      ourXcoord = math.floor((self.xSize - 2) / 2) - 2
    else:
      ourXcoord = math.floor((self.xSize - 2) / 2) + 2
    foodmatrix = []
    for i in range(16):
      if not gameState.hasWall(ourXcoord, i):
        foodmatrix.append((ourXcoord, i))
    return foodmatrix

  def getMinourzoneDist(self, gameState, red):
    foodmatrix = self.getourzone(gameState, red)
    mindistance = 999
    selfpos = gameState.getAgentPosition(self.index)
    for i in range(len(foodmatrix)):
      distance = self.getMazeDistance((foodmatrix[i][0], foodmatrix[i][1]), selfpos)
      if distance < mindistance:
        mindistance = distance
    return mindistance

  # updating weights (2)
  def updateWeights(self, gameState, action, difference):
    features = self.getFeatures(gameState, action)
    weights = self.weights
    alpha = 0.01
    
    for keys in weights.keys():
      weights[keys] = weights[keys] + difference * alpha * features[keys]

    return weights

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

  def getWeights(self, gameState, action):
    if (self.sentryEnabled):
      return self.sentryWeights
    else:
      return self.weights
  
  def getMinGhostDist(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if not gameState.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 44

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)

      mindistance = 99999
      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)

    return mindistance

  def getClosestGhostDist(self, successor, gameState):
    ghostmatrix = []
    for i in self.getOpponents(gameState):
      if not gameState.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
    
    if not ghostmatrix:
      return 44

    selfpos = successor.getAgentPosition(self.index)

    mindistance = 4
    for i in range(len(ghostmatrix)):
      distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
      if distance < mindistance:
        mindistance = distance

    return mindistance
  
  def getScaredGhostDist(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if not gameState.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer > 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)

      mindistance = 99999
      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)

    return -mindistance * (len(noisyghostmatrix) + len(ghostmatrix) + 1) ** 3

  def getClosestPacmanDist(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if gameState.getAgentState(i).isPacman and not successor.getAgentState(self.index).scaredTimer > 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)

      mindistance = 99999
      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)

    return -mindistance
  
  def minSuperPalletDist(self, successor, gameState):
    foodmatrix = self.superfoodMat
    mindistance = 220
    selfpos = gameState.getAgentPosition(self.index)
    for i in range(len(foodmatrix)):
      distance = self.getMazeDistance((foodmatrix[i][0], foodmatrix[i][1]), selfpos)
      if distance < mindistance:
        mindistance = distance
    return (self.getClosestGhostDist(successor, gameState) <= 5) * (220 - mindistance)

  def getDistEachOther(self, gameState, successor):
    meandist = 0
    teamlst = self.getTeam(gameState)
    for x in teamlst:
      agentpos = successor.getAgentPosition(x)
      meandist += self.getMazeDistance((agentpos[0], agentpos[1]), successor.getAgentPosition(self.index))
    meandist /= len(teamlst)
    return meandist

  def getHomeDist(self, gameState, successor):
    ourzones = self.getourzone(gameState, self.red)
    foodmatrix = ourzones[:]
    lastY = -1
    for i in ourzones:
      if i[1] == lastY+1:
        foodmatrix.remove(i)
      else:
        lastY = i[1]
    foodmatrix = self.getourzone(gameState, self.red)
    offset = math.floor(len(foodmatrix) / 3)
    distance = self.getMazeDistance(foodmatrix[offset], successor.getAgentPosition(self.index))
    return distance

  def getmyTarget(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if successor.getAgentState(i).isPacman and not successor.getAgentState(self.index).scaredTimer > 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    if len(ghostmatrix) == 2:
      i = 1 # chase enemy agent 1
    else:
      i = 0

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)
      mindistance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
    else:
      mindistance = noisyghostmatrix[i]

    return -mindistance * (len(ghostmatrix) + 1) ** 3

  def getEnemyPacmanDist(self, successor, gameState):
    # THIS IS DEFAULT AGENT!
    ghostmatrix = []
    noisyghostmatrix = []

    if len(ghostmatrix) + len(noisyghostmatrix) == 2:
      i = i = self.getOpponents(successor)[1] # chase enemy agent 1
    else:
      i = i = self.getOpponents(successor)[0]

    if gameState.getAgentState(i).isPacman and successor.getAgentState(self.index).scaredTimer <= 0:
      temp = gameState.getAgentPosition(i)
      if temp != None:
        ghostmatrix.append(temp)
      else:
        noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 44

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)

      mindistance = 99999
      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)

    return -mindistance * (len(noisyghostmatrix) + len(ghostmatrix) + 1) ** 5

  def getEatenGhost(self, successor, gameState):
    nowGhost = len([not gameState.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0 for i in self.getOpponents(gameState)])
    successorGhost = len([not gameState.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0 for i in self.getOpponents(successor)])
    return nowGhost > successorGhost

  def getEatenPacman(self, successor, gameState):
    nowGhost = len([gameState.getAgentState(i).isPacman and successor.getAgentState(self.index).scaredTimer <= 0 for i in self.getOpponents(gameState)])
    successorGhost = len([gameState.getAgentState(i).isPacman and successor.getAgentState(self.index).scaredTimer <= 0 for i in self.getOpponents(successor)])
    return nowGhost > successorGhost  

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    if self.sentryEnabled:
      features['home'] = self.getHomeDist(gameState, successor) / 220
      features['closestEnemyPacman'] = self.getmyTarget(successor, gameState) / 220
      features['sentrymodebonus'] = 1
      features['PacmanDist'] = self.getEnemyPacmanDist(gameState, successor) / 220
    else:
      features['successorScore'] = self.getScore(successor) / 5
      features['closestPalletDistance'] = (5 - gameState.getAgentState(self.index).numCarrying) * self.getMinPalletDist(gameState, successor) / 220
      features['foodCarrying'] = successor.getAgentState(self.index).numCarrying / 5
      features['shouldReturn'] = (gameState.getAgentState(self.index).numCarrying) * self.getMinourzoneDist(successor, self.red) / 220
      features['minGhostDist'] = self.getMinGhostDist(successor, gameState) / 220
      #features['scaredGhostDist'] = self.getScaredGhostDist(successor, gameState) / 220
      features['closeGhostDist'] = self.getClosestGhostDist(successor, gameState) / 44 - 1# 얘가 보너스 느낌이라 팰릿을 안먹었음
      features['superPallet'] = self.minSuperPalletDist(successor, gameState) / 220 # chase capsule when chasing
      features['closestEnemyPacman'] = self.getClosestPacmanDist(successor, gameState) / 220
      features['eatenGhost'] = self.getEatenGhost(successor, gameState)
      features['eatenPacman'] = self.getEatenPacman(successor, gameState)

    #features['distEachOther'] = self.getDistEachOther(gameState, successor) / 100

    # numCarrying이 5일때 따로 행동 지정?
    # 가장 가까운 우리 지역 위치를 지정?
    # 내가 팩맨일때, 그렇지 않을 떄 따로 지정?
    # 내가 파워 팰릿을 먹었을 때, 상대가 먹었을 때 등등...
    return features
  
  def initializeWeights(self):
    weights = util.Counter()
    weights['successorScore'] = 34.70 
    weights['closestPalletDistance'] = -7.18
    weights['foodCarrying'] = 8.13
    weights['shouldReturn'] = -14.175
    weights['minGhostDist'] = 3
    #weights['scaredGhostDist'] = 32.26
    weights['closeGhostDist'] = 38.53
    weights['closestEnemyPacman'] = 47.295
    weights['superPallet'] = 11.49
    weights['eatenGhost'] = 30
    weights['eatenPacman'] = 100
    #weights['distEachOther'] = 10
    return weights

  def getSentryWeights(self):
    weights = util.Counter()
    weights['closestEnemyPacman'] = 60
    weights['home'] = -1
    weights['PacmanDist'] = 50
    weights['sentrymodebonus'] = 100
    #weights['distEachOther'] = 10
    return weights

  def __init__( self, index, timeForComputing = .1 ):
    """
    Lists several variables you can query:
    self.index = index for this agent
    self.red = true if you're on the red team, false otherwise
    self.agentsOnTeam = a list of agent objects that make up your team
    self.distancer = distance calculator (contest code provides this)
    self.observationHistory = list of GameState objects that correspond
        to the sequential order of states that have occurred so far this game
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    """
    self.index = index
    self.red = None
    self.agentsOnTeam = None
    self.distancer = None
    self.observationHistory = []
    self.timeForComputing = timeForComputing
    self.display = None
    # some new stuff
    self.lastQ = 0
    self.lastaction = None
    self.weights = self.initializeWeights()
    self.foodMat = []
    self.superfoodMat = []
    self.sentryEnabled = False
    self.sentryWeights = self.getSentryWeights()

    self.xSize = 0
    self.ySize = 0
    

class Agent1(ReflexCaptureAgent):
  def getMinPalletDist(self, predecessor, gameState):
    foodmatrix = self.foodMat
    mindistance = 999
    chaseRemainder = True
    selfpos = gameState.getAgentPosition(self.index)
    for i in range(len(foodmatrix)):
      if foodmatrix[i][1] >= 9:
        chaseRemainder = False
        break
    for i in range(len(foodmatrix)):
      if foodmatrix[i][1] >= 9 or chaseRemainder: # run for upper pallets
        distance = self.getMazeDistance((foodmatrix[i][0], foodmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    return mindistance

  def getHomeDist(self, gameState, successor):
    ourzones = self.getourzone(gameState, self.red)
    foodmatrix = ourzones[:]
    lastY = -1
    for i in ourzones:
      if i[1] == lastY+1:
        foodmatrix.remove(i)
      else:
        lastY = i[1]

    offset = math.floor(len(foodmatrix) / 3)
    if (len(foodmatrix) == 2):
      offset = 1

    distance = self.getMazeDistance(foodmatrix[-offset], successor.getAgentPosition(self.index))
    return distance

  def getmyTarget(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if successor.getAgentState(i).isPacman and not successor.getAgentState(self.index).scaredTimer > 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    i = 0

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)
      mindistance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
    else:
      mindistance = noisyghostmatrix[i]

    return -mindistance * (len(ghostmatrix) + 1) ** 5

  def getEnemyPacmanDist(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []

    i = self.getOpponents(successor)[0]

    if gameState.getAgentState(i).isPacman and successor.getAgentState(self.index).scaredTimer <= 0:
      temp = gameState.getAgentPosition(i)
      if temp != None:
        ghostmatrix.append(temp)
      else:
        noisyghostmatrix.append(successor.getAgentDistances()[i])

    mindistance = 99999

    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    if len(ghostmatrix) > 0:
      selfpos = successor.getAgentPosition(self.index)

      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)
    
    return -mindistance * (len(noisyghostmatrix) + len(ghostmatrix) + 1) ** 5


class Agent2(ReflexCaptureAgent):
  def getMinPalletDist(self, predecessor, gameState):
    foodmatrix = self.foodMat
    mindistance = 999
    chaseRemainder = True
    selfpos = gameState.getAgentPosition(self.index)
    for i in range(len(foodmatrix)):
      if foodmatrix[i][1] < 9:
        chaseRemainder = False
        break
    for i in range(len(foodmatrix)):
      if foodmatrix[i][1] < 9 or chaseRemainder: # run for upper pallets
        distance = self.getMazeDistance((foodmatrix[i][0], foodmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    return mindistance

  def getHomeDist(self, gameState, successor):
    ourzones = self.getourzone(gameState, self.red)
    temp = ourzones[:]
    lastY = -1
    for i in ourzones:
      if i[1] == lastY+1:
        temp.remove(i)
      else:
        lastY = i[1]

    offset = math.floor(len(temp) / 3)
    foodmatrix = temp[:]

    for i in temp:
      if (self.getMazeDistance(temp[offset], i) <= 5 and temp[offset] != i):
        foodmatrix.remove(i)
    
    offset = math.floor(len(foodmatrix) / 3)

    distance = self.getMazeDistance(foodmatrix[offset], successor.getAgentPosition(self.index))
    return distance

  def getmyTarget(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []
    for i in self.getOpponents(gameState):
      if successor.getAgentState(i).isPacman and not successor.getAgentState(self.index).scaredTimer > 0:
        temp = gameState.getAgentPosition(i)
        if temp != None:
          ghostmatrix.append(temp)
        else:
          noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 0

    if len(ghostmatrix) == 2:
      i = 1 # chase enemy agent 1
    else:
      i = 0

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)
      mindistance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
    else:
      mindistance = noisyghostmatrix[i]

    return -mindistance * (len(ghostmatrix) + 1) ** 5

  def getEnemyPacmanDist(self, successor, gameState):
    ghostmatrix = []
    noisyghostmatrix = []

    if len(ghostmatrix) + len(noisyghostmatrix) == 2:
      i = i = self.getOpponents(successor)[1] # chase enemy agent 1
    else:
      i = i = self.getOpponents(successor)[0]

    if gameState.getAgentState(i).isPacman and successor.getAgentState(self.index).scaredTimer <= 0:
      temp = gameState.getAgentPosition(i)
      if temp != None:
        ghostmatrix.append(temp)
      else:
        noisyghostmatrix.append(successor.getAgentDistances()[i])
    
    if len(ghostmatrix) + len(noisyghostmatrix) == 0:
      return 44

    if ghostmatrix:
      selfpos = successor.getAgentPosition(self.index)

      mindistance = 99999
      for i in range(len(ghostmatrix)):
        distance = self.getMazeDistance((ghostmatrix[i][0], ghostmatrix[i][1]), selfpos)
        if distance < mindistance:
          mindistance = distance
    else:
      mindistance = min(noisyghostmatrix)

    return -mindistance * (len(noisyghostmatrix) + len(ghostmatrix) + 1) ** 5
