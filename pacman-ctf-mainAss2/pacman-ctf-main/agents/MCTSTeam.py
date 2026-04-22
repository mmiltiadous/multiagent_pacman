
from agents.MCTSAgents import OffensiveAgent, DefensiveAgent

def createTeam(
    firstIndex, secondIndex, isRed, first="OffensiveAgent", second="DefensiveAgent"
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

