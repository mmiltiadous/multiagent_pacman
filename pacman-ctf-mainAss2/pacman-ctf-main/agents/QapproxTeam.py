#Should be in the same folder with a folder named policyFolder which contains fourtxt files with names: 
#DefensiveQ, OffensiveQ, savebufferDefensiveQ, savebufferOffensiveQ
from captureAgents import CaptureAgent
from random import sample
from game import Directions
import random, util
import json
import os
# Create team and Agents

def createTeam(firstIndex, secondIndex, isRed, first = 'OffensivePlayer', second = 'DefensivePlayer'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  """
  locationSearching = Searching()
  locationSearching.__init__()

  return [eval(first)(firstIndex, locationSearching), eval(second)(secondIndex, locationSearching)]
 
# REINFORCEMENT LEARNING -- DQN 

class QApproximation(CaptureAgent):

    def __init__(self, index, locationSearching, numTraining=1000, learningRate=0, gamma=1, epsilon=0, waitTime=0.1,): #testing
    # def __init__(self, index, locationSearching, numTraining=1000, learningRate=0.6, gamma=1, epsilon=0.5, waitTime=0.1,):  #training 
        '''
        numTraining: number of training games 
        learningRate: learning rate, gamma: discount factor, epsilon: exploration rate
        '''
        CaptureAgent.__init__(self, index, waitTime)

        self.locationSearching = locationSearching
        self.numTraining = int(numTraining)
        self.Alpha = float(learningRate)
        self.Gamma0 = float(gamma) 
        self.Erate = float(epsilon)
        self.trainRaccum = float(0)
        self.exploitRaccum = float(0)
        self.completeEpisodes = int(1)
        self.weights = util.Counter()
        self.dict = util.Counter()
        self.buffer = []
    
    def registerInitialState(self, gameState):
        """
        Keeps track of games played to give as output(overwrite function from CaptureAgents.py)
        """
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.startEpisode()
        if self.completeEpisodes == 0:
            print('Starting training for %d episodes' % (self.numTraining))
    
    
    def getRewards(self, gameState, OffenceOn = True):
        """
        Take rewards based on 4 actions:
        self.deaths,
        self.scoreDenoted, 
        self.killPacman,
        self.foodtaken,
        move near to food
        """
        
        reward = 0
        #deaths
        initial_position = gameState.getInitialAgentPosition(self.index)
        current_position = gameState.getAgentPosition(self.index)
        final_position = self.final_state.getAgentPosition(self.index)
        if current_position == initial_position:
            finalX, finalY = final_position
            currentX, currentY = current_position
            if abs(finalX - currentX) > 1 or abs(finalY - currentY) > 1:
                reward += self.deaths
        #scoreDenoted
        current_score = self.getScore(gameState)
        final_score = self.final_state.getScore()
        if current_score > final_score:
            reward += current_score - final_score + self.scoreDenoted
        #killPacman
        opponents_before = [self.final_state.getAgentState(opp) for opp in self.getOpponents(self.final_state)]
        opponents_now = [gameState.getAgentState(opp) for opp in self.getOpponents(gameState)]
        pac_before = [a for a in opponents_before if a.isPacman and a.getPosition() != None]
        pac_now = [a for a in opponents_now if a.isPacman and a.getPosition() != None]
        num_pac_before = len(pac_before)
        if not (num_pac_before <= 0):
            distances=[self.getMazeDistance(self.final_state.getAgentState(self.index).getPosition(), a.getPosition()) for a in pac_before]
            if min(distances) == 1:
                agent_position_changed = gameState.getAgentState(self.index).getPosition() != gameState.getInitialAgentPosition(self.index)
                if len(pac_now) == 0 and agent_position_changed:
                    reward += self.killPacman
                elif len(pac_now) > 0 and agent_position_changed:
                    distances=[self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in pac_before]
                    if min(distances) > 2:
                        reward += self.killPacman
        #foodtaken
        current_food = self.getFood(gameState).asList()
        food_before = self.getFood(self.final_state).asList()
        current_quantiy = len(current_food)
        before_quantity = len(food_before)
        if not(current_quantiy <= before_quantity):
            reward += current_quantiy - before_quantity + self.foodtaken
        if OffenceOn:
        #move near to food
            current_pos = gameState.getAgentState(self.index).getPosition()
            final_pos = self.final_state.getAgentState(self.index).getPosition()
            current_food = self.getFood(gameState).asList()
            plusR = 0.5
            food = 0
            minimum_distance = 10000
            for cf in current_food:
                distance = self.getMazeDistance(current_pos, cf)
                if distance < minimum_distance:
                    minimum_distance = distance
                    food = cf
            if not (current_quantiy <= 0):
                RedYes = gameState.isOnRedTeam(self.index)
                current_X = current_pos[0]
                current_md = self.getMazeDistance(current_pos, food)
                final_md = self.getMazeDistance(final_pos, food)
                food_w = gameState.data.food.width
                if RedYes:
                    if current_md < final_md and current_X < food_w/2:
                        reward += plusR
                else:
                    if current_md < final_md and current_X > food_w/2:
                        reward += plusR
        return reward

    def getWeights(self):
        return self.weights
    
    def getBuffer(self):
        return self.buffer

    def startEpisode(self):
        self.current_reward = float(0)
        self.final_action = None
        self.final_state = None

    def return_qvalue(self, gameState, action):
        Q_v = 0
        succesor = gameState.generateSuccessor(self.index, action)
        features = self.locationSearching.getFeatures(succesor,self)
        Q_v = [features[feature] * self.weights[feature] for feature in features]
        totalQ_v = sum(Q_v)
        return totalQ_v
    
    
    def maximum_qv(self, gameState):
        """
        Calculates max Q-value among possible actions.
        """
        q_values = [self.return_qvalue(gameState, action) for action in gameState.getLegalActions(self.index)]
        return max(q_values)
    
    def Qaction(self, gameState):
        """
        Return best action, or none if no one available
        """
        legal_actions = [legact for legact in gameState.getLegalActions(self.index) if legact != Directions.STOP]
        bast_legact = None
        bestq = float(0)
        for legact in legal_actions:
            qv = self.return_qvalue(gameState, legact)
            if qv > bestq or bast_legact is None:
                bestq = qv
                bast_legact = legact
        return bast_legact

    def chooseAction(self, gameState):
        """
        Decide the next action randomly with probability epsilon or the best one otherwise. 
        If no actions availabe --> return NONE.
        Saves game state and reward to final_state and final_action respectively.
        """
        if self.final_state != None:
            self.update(gameState, self.getRewards(gameState))

            #update food
            myinfo_food = self.getFoodYouAreDefending(gameState).asList()
            final_info_food = self.getFoodYouAreDefending(self.final_state).asList()
            points_taken = [item for item in final_info_food if item not in myinfo_food]
            if len(points_taken) > 0:
              self.food_eaten_recently = points_taken[0]
              

        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return None
        legal_actions = [action for action in legal_actions if action != Directions.STOP]
        if not util.flipCoin(self.Erate):
            self.locationSearching.update_positions(gameState, self)
            action = self.Qaction(gameState)
        else:
            action = random.choice(legal_actions)
        self.y = gameState.data.food.height
        self.x = gameState.data.food.width
        self.final_state, self.final_action = gameState, action
        #check buffer
        if len(self.buffer) ==  1000:
            self.buffer.pop(0)
        return action

    def update(self, gameState, reward):
        """
        1)Calculates maximum q value and reward for the buffer.
        2)Average q-value and reward of 100 randomly chosen samples from buffer 
        3)Calculates difference of previousand average q-value 
        4)Update weights based on approximate formula of q-value : weight = weight + learning_rate * feature * diff
        5)Rubber feature to the weights (dont go out of proportion).
        """
        buffer_length = len(self.buffer)
        past_qv = self.return_qvalue(self.final_state, self.final_action)
        features = self.locationSearching.getFeatures(gameState,self)
        threshold = 12
        constant = 0.004
        diff = 0
        if len(gameState.getLegalActions(self.index)) != 0:
            maximum_qv = self.maximum_qv(gameState)
            if not (self.Alpha <= 0):
              self.buffer.append((maximum_qv, reward))
            averages = sample(self.buffer, int(buffer_length/threshold))
            total = self.Gamma0*maximum_qv
            for a in averages:
                total += a[1] + self.Gamma0*a[0]
            total = total/(len(averages) + 1)
            total = reward + total
            diff = total - past_qv
        else:
            diff =  reward - past_qv
        for f in features:
            update_weight = self.weights[f] + self.Alpha * features[f] * diff
            #Thresholding
            if update_weight > threshold:
                update_weight = min(update_weight, threshold)
            elif update_weight < -threshold:
                update_weight = max(update_weight, -threshold)
            self.weights[f] = update_weight
            band = constant*self.Alpha
            #Band 
            if self.weights[f] > 0:
                self.weights[f] = max(0, update_weight - band)
            elif self.weights[f] < 0:
                self.weights[f] = min(0, update_weight + band)

    
    def save_policy(self, AgentPolicy):
        """
        Writes existing policy to a policyFolder
        """
        f1= open("./policyFolder/"+AgentPolicy,"w+")
        d1 = json.dumps(self.getWeights())
        f1.write(d1)
        f1.close()

        f2 = open("./policyFolder/savebuffer"+AgentPolicy,"w+")
        d2 = json.dumps(self.getBuffer())
        f2.write(d2)
        f2.close()

    def ReadPolicy(self, AgentPolicy):
        """
        Reads the policy if exists in policyFolder or creates new one if does not
        """
        if os.stat("./policyFolder/"+AgentPolicy).st_size == 0:
          self.weights = util.Counter()
        else:
          self.weights = util.Counter()
          policyread = open("./policyFolder/"+AgentPolicy, 'r').read()
          policy_dictionary = json.loads(policyread)
          for feature in policy_dictionary:
            self.weights[feature] = policy_dictionary[feature]
        
        if os.stat("./policyFolder/savebuffer"+AgentPolicy).st_size == 0:
          self.buffer = []
        else:
          self.buffer = util.Counter()
          buffer_read = open("./policyFolder/savebuffer"+AgentPolicy, 'r').read()
          parsedBuffer = json.loads(buffer_read)
          self.buffer = parsedBuffer

# Defensive and Offensive Agents

class DefensivePlayer(QApproximation):
    """
    Defensive Agent 
    Overwrites getRewards() and final() fun
    """

    def __init__(self, index, locationSearching):
        QApproximation.__init__(self, index, locationSearching)
        self.deaths = -20
        self.scoreDenoted = 15
        self.killPacman = 20
        self.foodtaken = 7
        self.ReadPolicy("DefensiveQ.txt")
    
    def getRewards(self, gameState, OffenceOn = True):
        """
        Take rewards based on 4 actions:
        self.deaths,
        self.scoreDenoted, 
        self.killPacman,
        self.foodtaken
        """
        return QApproximation.getRewards(self, gameState, OffenceOn = False)
    
    def final(self, gameState):
        """
        Store policy
        """
        QApproximation.final(self, gameState)
        self.save_policy("DefensiveQ.txt")


class OffensivePlayer(QApproximation):
    """
    Offsenvie Agent
    Overwrites getRewards() and final() fun
    """

    def __init__(self, index, locationSearching):
        QApproximation.__init__(self, index, locationSearching)
        self.deaths = -20
        self.scoreDenoted = 15
        self.killPacman = 5
        self.foodtaken = 12
        self.ReadPolicy("OffensiveQ.txt")
    
    
    def getRewards(self, gameState, OffenceOn = True):
        """
        Take rewards based on 4 actions:
        self.deaths,
        self.scoreDenoted, 
        self.killPacman,
        self.foodtaken,
        move near to food
        """
        return QApproximation.getRewards(self, gameState)
    

    def final(self, gameState):
        """
        Store policy
        """
        QApproximation.final(self, gameState)
        self.save_policy("OffensiveQ.txt")

# Helping class

class Searching:

  def __init__(self):
    self.pacman_position = []
    self.ghost_position = []
    self.ghosts_st = []
    self.info_food = []
    self.food1 = 0
    self.food2 = 0
    self.x = 0
    self.y = 0
    self.food_eaten_recently = None


  def update_positions(self, gameState, agent):
    opponents = [opp for opp in agent.getOpponents(gameState)]
    opponents_pacman = [gameState.getAgentPosition(opp) for opp in opponents if gameState.getAgentState(opp).isPacman and gameState.getAgentPosition(opp) is not None]
    ghosts = [gameState.getAgentPosition(opp) for opp in opponents if not gameState.getAgentState(opp).isPacman and gameState.getAgentPosition(opp) is not None]
    self.ghosts_st = [gameState.getAgentState(opp) for  opp in opponents if not gameState.getAgentState(opp).isPacman and gameState.getAgentPosition(opp) is not None]
    self.ghost_position = ghosts
    self.pacman_position = opponents_pacman
    self.myinfo_food = agent.getFoodYouAreDefending(gameState).asList()
    self.info_food = agent.getFood(gameState).asList()
    a_ind = agent.index
    if a_ind == 0 or a_ind == 1:
      self.food1 = gameState.getAgentState(a_ind).numCarrying
    else:
      self.food2 = gameState.getAgentState(a_ind).numCarrying

  
  def dist_opponent(self, gameState, agent, choice):
    """
    Calculate distance from the closest opponent agent (ghost or Pacman).
    """
    if choice == 1:  # Ghost
        decision = 0
        for ghost in self.ghosts_st:
            if ghost.scaredTimer > 10:  # Ignore if scared
                for pos in self.ghost_position:
                    if pos == ghost.getPosition():
                        self.ghost_position.remove(pos)
            elif ghost.scaredTimer > 3:  # Attack if scared
                decision = 1
        if len(self.ghost_position) > 0:
            distances = [agent.getMazeDistance(gameState.getAgentState(agent.index).getPosition(), pos) for pos in self.ghost_position]
            smallest_distance = min(distances)
            if smallest_distance > 5:
                smallest_distance *= 2
            if smallest_distance == 0:
                smallest_distance = 0.5
            if decision == 1:
                smallest_distance = -smallest_distance
            return 1 / smallest_distance if smallest_distance != 0 else 0
        else:
            return 0
    elif choice == 0:  # Pacman
        pac_pos_=self.pacman_position
        if len(pac_pos_) > 0:
            distances = [agent.getMazeDistance(gameState.getAgentState(agent.index).getPosition(), pos)for pos in pac_pos_]
            smallest_distance = min(distances)
            if smallest_distance == 0:
                smallest_distance = 0.5
            if gameState.getAgentState(agent.index).scaredTimer > 0 and 0.5 <= smallest_distance <= 2:
                # Safe from scared ghost
                smallest_distance = -smallest_distance
            return 1 / smallest_distance if smallest_distance != 0 else 0
        else:
            return 0
    return 0

  def dist_from_teamside(self, gameState, agent, count): 
    """
    Calculates minimum distance from the side. Assigns a large number if there's a ghost on the path.
    """
    big_num = 1000
    count_thresh = 100
    small_num = 0.5
    legalActions = gameState.getLegalActions(agent.index)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    count += 1
    if count >= count_thresh:
        return 0
    my_position = gameState.getAgentState(agent.index).getPosition()
    initial_position = gameState.getInitialAgentPosition(agent.index)
    if not gameState.getAgentState(agent.index).isPacman and my_position != initial_position:
        return small_num
    if my_position == initial_position:
        return big_num
    next_action = legalActions[0]
    smallest_distance = big_num
    for l_action in legalActions:
        successor = gameState.generateSuccessor(agent.index, l_action)
        new_location = successor.getAgentState(agent.index).getPosition()
        distance = agent.getMazeDistance(new_location, initial_position)
        if distance < smallest_distance:
            next_action = l_action
            smallest_distance = distance
    successor = gameState.generateSuccessor(agent.index, next_action)
    distance_from_border = self.dist_from_teamside(successor, agent, count) + 1
    return distance_from_border
  
  def getFeatures(self, gameState, agent):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    if agent.index == 0 or agent.index == 1: #offensive
        #closest food
        min_distance = 0 if len(self.info_food) < 3 else min([agent.getMazeDistance(gameState.getAgentState(agent.index).getPosition(), food) for food in self.info_food])
        food_close = 2 if min_distance == 0 else (1 / min_distance if min_distance != 0 else 0)
        features['food_close'] = food_close
        #points carryed by agent
        #food carryed
        carry_weight = self.food1 if agent.index == 0 or agent.index == 1 else self.food2
        points_on_agent = 0 if carry_weight == 0 else (1 / self.dist_from_teamside(gameState, agent, 0)) * carry_weight
        features['points_on_agent'] = points_on_agent
        #ghosts close
        features['ghosts_close'] = self.dist_opponent(gameState, agent, 1)
    else:  #defensve
        #food eaten close 
        distance_ = 1
        food_eaten_close = None
        if self.food_eaten_recently == None: #if nothing eaten, gives the most away position
            myinfo_food = agent.getFoodYouAreDefending(gameState).asList()
            initial_most_away = 0
            for food_i in myinfo_food:
                most_away = 0
                distance_ToFood = agent.getMazeDistance(gameState.getInitialAgentPosition(agent.index), food_i)
                if distance_ToFood > most_away:
                    initial_most_away = food_i
                    most_away = distance_ToFood
            distance_ = agent.getMazeDistance(gameState.getAgentState(agent.index).getPosition(), initial_most_away)
            if distance_ <= 3:  # within 3 moves:patrol area more
                distance_ = 1
                food_eaten_close = 1
            else:
                food_eaten_close = 1/distance_
        if food_eaten_close is None:
            distance_ = agent.getMazeDistance(gameState.getAgentState(agent.index).getPosition(), self.food_eaten_recently)
            if distance_ <= 3:
                distance_ = 1
                food_eaten_close = 1
            else:
                food_eaten_close = 1/distance_
        features['food_eaten_close'] = food_eaten_close
        #ghost close
        features['ghosts_close'] = self.dist_opponent(gameState, agent, 1)
    #pacmanclose
    features['pacman_close'] = self.dist_opponent(gameState, agent, 0)
    return features
  