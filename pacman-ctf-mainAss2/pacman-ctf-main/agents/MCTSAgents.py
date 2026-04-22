import random, util

from captureAgents import CaptureAgent
from game import Directions

from agents.MCTSNode import MCTSNode


class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)


    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        if agent_state.isPacman:
            enemy_list = [
                enemy
                for enemy in self.getOpponents(gameState)
                if not gameState.getAgentState(enemy).isPacman
                and gameState.getAgentState(enemy).scaredTimer == 0
                and gameState.getAgentPosition(enemy) is not None
            ]
            dangerGhosts = [
                g
                for g in enemy_list
                if self.getMazeDistance(
                    gameState.getAgentPosition(self.index),
                    gameState.getAgentPosition(g),
                )
                <= 10
            ]
            appr_ghost_pos = [gameState.getAgentPosition(g) for g in dangerGhosts]

            if not appr_ghost_pos:
                values = [
                    self.get_offense_features(gameState, action)
                    * self.get_offense_weights()
                    for action in actions
                ]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                action_chosen = random.choice(bestActions)
            else:
                rootNode = MCTSNode(
                    gameState,
                    self,
                    None,
                    None,
                    appr_ghost_pos,
                    self.detect_friendly_boarder(gameState),
                )
                action_chosen = MCTSNode.mcts_search(rootNode)

        else:
            values = [
                self.get_defense_feat(gameState, action) * self.get_defense_w()
                for action in actions
            ]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            action_chosen = random.choice(bestActions)

        return action_chosen


    def detect_friendly_boarder(self, gameState):

        if self.red:
            border_x = int(gameState.data.layout.width / 2) - 1
        else:
            border_x = int(gameState.data.layout.width / 2)
        border_line = [(border_x, h) for h in range(gameState.data.layout.height)]
        return [
            (x, y)
            for (x, y) in border_line
            if (x, y) not in gameState.getWalls().asList()
            and (x + 1 - 2 * self.red, y) not in gameState.getWalls().asList()
        ]


    def get_defense_feat(self, gameState, action):

        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()
        features["successor_score"] = -len(foodList)

        current_pos = successor.getAgentState(self.index).getPosition()
        features["distance_to_food"] = min(
            [self.getMazeDistance(current_pos, food) for food in foodList]
        )
        return features


    def get_defense_w(self):

        return {"successor_score": 100, "distance_to_food": -1}


    def get_offense_features(self, gameState, action):

        features = util.Counter()
        next_state = gameState.generateSuccessor(self.index, action)
        if (
            next_state.getAgentState(self.index).numCarrying
            > gameState.getAgentState(self.index).numCarrying
        ):
            features["get_food"] = 1
        else:
            if len(self.getFood(next_state).asList()) > 0:
                features["min_dist_food"] = min(
                    [
                        self.getMazeDistance(
                            next_state.getAgentState(self.index).getPosition(), f
                        )
                        for f in self.getFood(next_state).asList()
                    ]
                )
        return features


    def get_offense_weights(self):

        return {"min_dist_food": -1, "get_food": 100}



class DefensiveAgent(OffensiveAgent):

    def get_defense_feat(self, gameState, action):
        features = util.Counter()
        next_state = gameState.generateSuccessor(self.index, action)

        my_state = next_state.getAgentState(self.index)
        my_pos = my_state.getPosition()

        features["defending"] = 1
        if my_state.isPacman:
            features["defending"] = 0

        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]
        features["n_inv"] = len(invaders)
        if len(invaders) > 0:
            features["inv_distance"] = min(
                [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            )

        if action == Directions.STOP:
            features["stopping"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reversing"] = 1

        return features


    def get_defense_w(self):
        return {
            "n_inv": -1000,
            "defending": 100,
            "inv_distance": -10,
            "stopping": -100,
            "reversing": -2,
        }
