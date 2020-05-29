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
import random
import math
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='OffensiveReflexAgent'):
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
    # gameSTate
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# 왼쪽아래가 0,0

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    ave_posBelief = [util.Counter(),util.Counter(),util.Counter(),util.Counter()]
    ave_cnt = 0
    cnt = 0
    target = [(0,0),(0,0),(0,0),(0,0)]

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # myinit
        self.myinitMAP(gameState)
        self.myinitPosBelief(gameState, 1)
        self.enemVisible = 0
        self.enemSeen = -1
        # print(self.mygetProbNoise())

    def chooseAction(self, gameState):
        # if gameState.getAgentPosition(self.index) == self.start: util.pause()

        self.enemVisible = 0        
        enemdistance =99999
        for enemy in self.getOpponents(gameState) :
            if(gameState.getAgentPosition(enemy) != None):
                self.enemVisible += 1
                aenemdistance = self.getMazeDistance(gameState.getAgentPosition(self.index),gameState.getAgentPosition(enemy))
                if(aenemdistance<enemdistance):
                    enemdistance = aenemdistance
                    self.enemSeen = enemy
        # if self.cnt % 20 == 0:
        #     util.pause()
        # self.cnt += 1
        # self.testmyMethods(gameState)
        # print(gameState.getAgentPosition(self.getOpponents(gameState)[0]))
        """
        Picks among the actions with the highest Q(s,a).
        """
        # self.myElapseTime(gameState)
        # self.myObserve(gameState)
        # self.myAvePosBelief(gameState, self.posBelief)
        # for enemy in self.getOpponents(gameState):
        # if self.ave_cnt % 2 == 0:
            # self.displayDistributionsOverPositions(self.ave_posBelief)
        actions = gameState.getLegalActions(self.index)
        # if 'Stop' in actions:
        #     actions.remove('Stop')
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        print(values)
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

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

########my method start#####################################################
#테스트
    # 내 메소드 테스트
    def testmyMethods(self, gameState):
        print("--agent"+str(self.index)+"--")
        print(self.mygetCarrying(gameState))
        print(self.mygetScore(gameState))
        print(self.mygetVisibleEnemy(gameState))
        print()
        return

#유틸
    # x,y 원점대칭
    def mygetPointSym(self, point):
        return (self.MAPWIDTH-point[0], self.MAPHEIGHT-point[1])

    # agent가 들고있는 food 갯수
    def mygetCarrying(self, gameState):
        return gameState.getAgentState(self.index).numCarrying

    # 현재 score
    def mygetScore(self, gameState):
        return self.getScore(gameState)

    # 가시거리에 있는 적agent위치
    def mygetVisibleEnemy(self, gameState):
        self.displayDistributionsOverPositions(gameState.getRedFood())
        return

    # N(0,9), 맨하탄dist에 있는 각 칸별 확률계산
    def mygetProbNoise(self):
        a = util.Counter()
        # def calc(dist): 
        #     return (math.erfc(-0.235702*(dist+0.5)) - math.erfc(-0.235702*(dist-0.5)))
        
        # a[0] = math.erfc(-0.235702*(0.5)) - math.erfc(-0.235702*(0))
        # for i in range(1,10):
        #     a[i] = calc(i)
        # a.normalize()
        # for i in range(1,10):
        #     a[i] /= i*4
        
        a[0] = 0.13257194469915404
        a[1] = 0.06273599146912161
        a[2] = 0.026593279695890325
        a[3] = 0.013463406341469558
        a[4] = 0.006868760175355754
        a[5] = 0.0033482387247029525
        a[6] = 0.0015228838321323362
        a[7] = 0.0006381638694748561
        a[8] = 0.00024452862582736865
        a[9] = 8.525894769175877e-05
        return a

#MAP
    def myinitMAP(self, gameState):
        #틀
        self.MAP = gameState.getWalls()
        self.MAPWIDTH = self.MAP.width
        self.MAPHEIGHT = self.MAP.height

        self.posNonWall = gameState.data.layout.walls.asList(False)
        # print(self.MAP.asList())
        # print(self.MAPWIDTH)

# Belief HMM 시작
    # Belief 초기화
    def myinitPosBelief(self, gameState, first = 0):
        self.posBelief = [util.Counter(),util.Counter(),util.Counter(),util.Counter()]
        # if first == 0:
        for enemy in self.getOpponents(gameState):
            for x in range(self.MAPWIDTH):
                for y in range(self.MAPHEIGHT):
                    if(self.MAP[x][y] == False):
                        self.posBelief[enemy][(x, y)] = 0.5
        # else:
        #     for enemy in self.getOpponents(gameState):
        #         print('first init')
        #         self.posBelief[enemy][self.mygetPointSym(self.start)] = 0.5
        return

    # 시간경화
    def myElapseTime(self, gameState):
        #일반적으로
        for enemy in self.getOpponents(gameState):
            newBelief = util.Counter()
            for (oldX, oldY), oldProb in self.posBelief[enemy].items():
                newProb = util.Counter()
                for p in [(oldX - 1, oldY), (oldX + 1, oldY), (oldX, oldY - 1), (oldX, oldY + 1), (oldX, oldY)]:
                    if p in self.posNonWall:
                        newProb[p] = 1.0
                newProb.normalize()
                for newPos, prob in newProb.items():
                    newBelief[newPos] += prob * oldProb #HMM 식
            newBelief.normalize()
            self.posBelief[enemy] = newBelief

        # 우리팀 음식 없어졌을시 (임시)
        thiefIndex = (self.index - 1) % 4
        lostFood = []
        lastObserved = self.getPreviousObservation()
        if lastObserved:
            lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList() if food not in self.getFoodYouAreDefending(gameState).asList()]
        if len(lostFood) != 0:
            f = lostFood[0]
            newBelief = util.Counter()
        # if self.eatenFood[f] != 1:
            # self.eatenFood[f] = 1
            newBelief[f] = 1.0
            self.posBelief[thiefIndex] = newBelief

        # 발견시 (임시)
        # 구현중

    # 위치추측(구현중)
    def guessPosition(self, agent, gameState):
        result = self.posBelief[agent].argMax()
        # if self.posBelief[agent][result] == 1:
        return result
        # else:
        #     return gameState.getInitialAgentPosition(agent)

    # 관측
    def myObserve(self, gameState):
        noisyDist = gameState.getAgentDistances() # + round(N(0,3^2)) ++ 2개 결과 합치는방법 없을까?
        myPos = gameState.getAgentPosition(self.index)
        teamPos = [gameState.getAgentPosition(team) for team in self.getTeam(gameState)]
        for enemy in self.getOpponents(gameState):
            newBelief = util.Counter()
            pos = gameState.getAgentPosition(enemy)
            if pos:
                if util.manhattanDistance(myPos, pos) < 6:
                    newBelief[(pos)] = 1.0
                    self.posBelief[enemy] = newBelief
            else:
                for p in self.posNonWall:
                    if self.red: #내가 고스트인 구역이면 1
                        isPac = p[0] < self.MAPWIDTH/2
                    else:
                        isPac = p[0] > self.MAPWIDTH/2
                    if any([util.manhattanDistance(teamPosition, p) <= 5 for teamPosition in teamPos]): #보이는 구역은 확실히 판별가능
                        newBelief[p] = 0.0
                    elif isPac != gameState.getAgentState(enemy).isPacman: #우리구역인데 고스트다? 그럼 아님
                        newBelief[p] = 0.0
                    else:
                        pDist = util.manhattanDistance(myPos, p)
                        absDist = abs(pDist - noisyDist[enemy])
                        posProb = self.mygetProbNoise()[absDist]
                        newBelief[p] = posProb * self.posBelief[enemy][p]
                if newBelief.totalCount() == 0:
                    self.myinitPosBelief(gameState)
                else:   
                    newBelief.normalize()
                    self.posBelief[enemy] = newBelief

    def myAvePosBelief(self, gameState, posBelief):
        a = 0
        for enemy in self.getOpponents(gameState):
            if self.ave_cnt % 2 == 0:
                self.ave_posBelief[enemy] = self.posBelief[enemy]
            else:
                for pos in self.ave_posBelief[enemy].keys():
                    self.ave_posBelief[enemy][pos] = ((self.ave_posBelief[enemy][pos] + self.posBelief[enemy][pos])/2)*(a)+ (math.sqrt(self.ave_posBelief[enemy][pos] * self.posBelief[enemy][pos]))*(1-a)
                # self.ave_posBelief[enemy] += self.posBelief[enemy]
                self.ave_posBelief[enemy].normalize()
        self.ave_cnt+=1

#
#########my method end####################################################

    def evaluate(self, gameState, action):

        # """
        # Computes a linear combination of features and feature weights
        # """
        # successor = self.getSuccessor(gameState, action)
        # actions = successor.getLegalActions(self.index)
        # if 'Stop' in actions:
        #     actions.remove('Stop')
        # def deep_eval(a):
        #     features = self.getFeatures(successor, a)
        #     weights = self.getWeights(successor, a)
        #     return features * weights

        # vals = [deep_eval(a) for a in actions]
        # max_vals = max(vals)
        # now = self.getFeatures(gameState,action)*self.getWeights(gameState,action)
        # bestActions = [v for v in vals if v == max_vals]
        # if len(bestActions) > 1:
        #     return now
        # return max(vals) + now
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
        features['remainFoodNum'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'remainFoodNum': 1.0}

#값들!!
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        #
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()

        midPoints = []
        for y in range(1, self.MAPHEIGHT):
            if(self.red):
                point = ((self.MAPWIDTH/2)-1,float(y))
            else:
                point = (self.MAPWIDTH/2,float(y))
            if point in self.posNonWall:
                midPoints.append(point)

        features['remainFoodNum'] = -len(foodList)  # self.getScore(successor)
        features['Carrying'] = self.mygetCarrying(successor)
        # features['goHome'] = self.mygetCarrying(successor)*int(action == Directions.WEST)
        features['goHome'] = self.mygetCarrying(successor)*min([self.getMazeDistance(myPos, dots) for dots in midPoints])
        features['Score'] = self.getScore(successor)
        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            # myPos = successor.getAgentState(self.index).getPosition()
            minDistance = 99999
            for food in foodList:
                if food != self.target[(self.index+2)%4]:
                    distances = self.getMazeDistance(myPos, food)
                    if distances < minDistance:
                        minDistance = distances
                        self.target[self.index] = food

        features['distanceToFood'] = minDistance
        
        features['foodTarget'] = 0
        if self.index == self.getTeam(gameState)[1]:
            if(self.target[(self.index+2)%4]) != (0,0):
                features['foodTarget'] =self.getMazeDistance(myPos, self.target[(self.index+2)%4])
        
        if features['Carrying'] == 5:
            features['distanceToFood'] = 0
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        features['getChased'] = 0
        features['enemDist'] = 0
        features['chasedgoHome'] = 0
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" + str(self.enemVisible))
        if self.enemVisible > 0 :
            enemdistance = self.getMazeDistance(myPos, successor.getAgentPosition(self.enemSeen))
            if (enemdistance < 3 and ((successor.getAgentState(self.index).isPacman) and (not(successor.getAgentState(self.enemSeen).isPacman))) or ((successor.getAgentState(self.index).scaredTimer>0) and successor.getAgentState(self.enemSeen).isPacman)):
                features['getChased'] = 1
                # if features['getChased']: util.pause()
                features['enemDist'] = enemdistance
                features['chasedgoHome'] = int(gameState.getAgentState(self.index).isPacman) * min([self.getMazeDistance(myPos, dots) for dots in midPoints])
                features['stop'] = 0
                features['reverse'] = 0
                


        # if not(len(self.observationHistory) < 4):
        #     posHis1 = self.observationHistory[-2].getAgentPosition(self.index)
        #     posHis2 = self.observationHistory[-3].getAgentPosition(self.index)
        #     posHis3 = self.observationHistory[-4].getAgentPosition(self.index)
        #     # rev1 = Directions.REVERSE[self.observationHistory[-2].getAgentState(self.index).configuration.direction]
        #     # rev2 = Directions.REVERSE[self.observationHistory[-3].getAgentState(self.index).configuration.direction]
        #     # rev3 = Directions.REVERSE[self.observationHistory[-4].getAgentState(self.index).configuration.direction]
        #     features['posHis1'] = int(successor.getAgentPosition(self.index) == posHis1)
        #     features['posHis2'] = int(successor.getAgentPosition(self.index) == posHis2)
        #     features['posHis3'] = int(successor.getAgentPosition(self.index) == posHis3)

        print('-------'+str(self.index)+action+'--------')
        print(features)
        return features

    def getWeights(self, gameState, action):
        return {'Score': 1000, 'remainFoodNum': 1751, 'distanceToFood': -95, 'Carrying' : 0,'goHome' : -23, 'stop' : -10000, 'reverse' : -10000, 
        'getChased' : -99999, 'enemDist' : +3000, 'chasedgoHome' : -100,  'foodTarget' : -10}


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
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,}
