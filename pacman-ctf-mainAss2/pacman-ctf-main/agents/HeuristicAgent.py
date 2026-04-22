from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="OffensiveHeuristicAgent",
    second="DefensiveHeuristicAgent",
):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class HeuristicAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

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
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features["successorScore"] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):

        return {"successorScore": 1.0}


class OffensiveHeuristicAgent(HeuristicAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features["successorScore"] = -len(foodList)

        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            Distances = []
            for food in foodList:
                Distances.append(self.getMazeDistance(myPos, food))

            features["distanceToFood"] = min(Distances)

        opponents = self.getOpponents(successor)
        invaders = []
        for a in self.getOpponents(successor):
            if successor.getAgentState(a).isPacman:
                invaders.append(a)
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            dists = []
            for invader in invaders:
                if successor.getAgentPosition(invader) !=None:
                    dists.append(
                        self.getMazeDistance(myPos, successor.getAgentPosition(invader))
                    )
                else: #returns random distance if successor.getAgentPosition(invader none
                    random_dist = random.randint(1, 50)
                    dists.append(
                        random_dist
                    )

            features["invaderDistance"] = min(dists)

        CapsuleDistance = []
        capsules = self.getCapsules(successor)
        if capsules:
            for capsule in capsules:
                CapsuleDistance.append(self.getMazeDistance(myPos, capsule))
            features["distanceToCapsule"] = min(CapsuleDistance)

        # Distance to teammate

        teammates = []
        for i in self.getTeam(successor):
            if i != self.index:
                teammates.append(successor.getAgentState(i))

        teammate_positions = []
        for a in teammates:
            if a.getPosition() is not None:
                teammate_positions.append(a.getPosition())
        TeammateDistance = []
        if teammate_positions:
            for teammate in teammate_positions:
                TeammateDistance.append(self.getMazeDistance(myPos, teammate))
            features["distanceToTeammate"] = min(TeammateDistance)
        return features

    def getWeights(self, gameState, action):
        return {
            "successorScore": 100,
            "distanceToFood": -2,
            "numInvaders": -1000,
            "invaderDistance": -10,
            "distanceToCapsule": -20,
            "distanceToTeammate": -5,
        }


class DefensiveHeuristicAgent(HeuristicAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        enemies = []
        for i in self.getOpponents(successor):
            enemies.append(successor.getAgentState(i))
        invaders = []
        for a in enemies:
            if a.isPacman and a.getPosition() != None:
                invaders.append(a)
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            dists = []
            for a in invaders:
                dists.append(self.getMazeDistance(myPos, a.getPosition()))
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        teammates = []
        for i in self.getTeam(successor):
            if i != self.index:
                teammates.append(successor.getAgentState(i))
        teammate_positions = []
        for a in teammates:
            if a.getPosition() is not None:
                teammate_positions.append(a.getPosition())

        TeammateDistance = []
        if teammate_positions:
            for teammate in teammate_positions:
                TeammateDistance.append(self.getMazeDistance(myPos, teammate))
            features["distanceToTeammate"] = min(TeammateDistance)

        foodList = self.getFoodYouAreDefending(successor).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features["distanceToFood"] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            "numInvaders": -1000,
            "onDefense": 100,
            "invaderDistance": -10,
            "stop": -100,
            "reverse": -2,
            "distanceToTeammate": -5,
            "distanceToFood": -1,
        }
