import random, time, util
import numpy as np


class MCTSNode:
    def __init__(
        self, game_state, agent, action, parent, enemy_pos, borderline, max_depth=15
    ):
        self.parent = parent
        self.action = action
        self.depth = parent.depth + 1 if parent else 0
        self.max_depth = max_depth

        self.children = []
        self.visits = 1
        self.q_value = 0.0

        self.game_state = game_state.deepCopy()
        self.enemy_pos = enemy_pos
        self.legal_actions = [
            action
            for action in game_state.getLegalActions(agent.index)
            if action != "Stop"
        ]
        self.unexplored_actions = self.legal_actions.copy()
        self.borderline = borderline

        self.agent = agent
        self.epsilon = 0.1
        self.rewards = 0

    def expand(self):
        if self.depth >= self.max_depth:
            return self

        if self.unexplored_actions:
            action = self.unexplored_actions.pop()
            current_game_state = self.game_state.deepCopy()
            next_game_state = current_game_state.generateSuccessor(
                self.agent.index, action
            )
            child_node = MCTSNode(
                next_game_state,
                self.agent,
                action,
                self,
                self.enemy_pos,
                self.borderline,
            )
            self.children.append(child_node)
            return child_node

        if util.flipCoin(self.epsilon):
            next_best_node = self.best_child()
        else:
            next_best_node = random.choice(self.children)

        if next_best_node is not None:
            return next_best_node.expand()
        else:
            return None

    def mcts_search(self):
        time_limit = 0.99
        start = time.time()
        while time.time() - start < time_limit:
            node_selected = self.expand()
            if node_selected is not None:
                if node_selected.game_state.getAgentPosition(
                    node_selected.agent.index
                ) == node_selected.game_state.getInitialAgentPosition(node_selected.agent.index):
                    reward = -1_000_000
                else:
                    reward = node_selected.get_feature() * node_selected.get_weight_backup()
                node_selected.visits += 1
                node_selected.q_value += reward
                if node_selected.parent is not None:
                    node_selected.parent.visits += 1
                    node_selected.parent.q_value += reward
            total_time = time.time() - start
        print('Total time for move',total_time)


        best_child = self.best_child()
        if best_child is not None:
            return best_child.action
        else:
            return None

    def best_child(self):
        best_score = -np.inf
        best_child = None
        for candidate in self.children:
            score = candidate.q_value / candidate.visits
            if score > best_score:
                best_score = score
                best_child = candidate
        return best_child


    def get_feature(self):
        feature = util.Counter()
        feature["distance"] = min(
            [
                self.agent.getMazeDistance(
                    self.game_state.getAgentPosition(self.agent.index), border_pos
                )
                for border_pos in self.borderline
            ]
        )
        return feature


    def get_weight_backup(self):
        return {"distance": -1}
