# Multi-Agent Pacman Capture the Flag
## Description

A multi-agent AI project built on the Berkeley Pacman Capture the Flag framework (Python 3 port by [cshelton](https://github.com/cshelton/pacman-ctf)). Two teams of agents compete to eat as many food pellets as possible from the opposing side while defending their own territory.
 
## Agents

Three agents were implemented, each using a different AI strategy:
1. QapproxTeam.py : Approximate Q-Learning (One offensive and one defensive agent that learn via function approximation Q-learning. Weights are trained over many episodes and saved to PolicyFolder/).
2. MCTSTeam.py : Monte Carlo Tree Search (Offensive agent uses MCTS when danger ghosts are nearby, falls back to a weighted heuristic otherwise. Defensive agent uses a heuristic policy).
3. HeuristicAgent.py : Rule-based Heuristic (Offensive and defensive agents guided by hand-crafted feature weights, like food distance, invader count, capsule distance, teammate distance, etc.).


## Folders/Files

- `agents/` : Implemented agent files
  - `QapproxTeam.py` : Approximate Q-Learning team
  - `MCTSTeam.py` : MCTS team
  - `MCTSAgents.py` : MCTS offensive/defensive agent logic
  - `MCTSNode.py` : MCTS node and search implementation
  - `HeuristicAgent.py` : Heuristic-based team
- `PolicyFolder/` : Trained Q-Learning weights and replay buffers (weights required at runtime)
  - `DefensiveQ.txt`
  - `OffensiveQ.txt`
  - `savebufferDefensiveQ.txt`
  - `savebufferOffensiveQ.txt`
- `results/` : Evaluation output CSVs (win/draw/lose counts)
- `agent_eval.py` : Script to evaluate agents over N games
- `capture.py` : Main game entry point (from pacman-ctf)
- `report.pdf` : Full project report
- `problem_description.pdf` : Assignment 

All other files (`game.py`, `captureAgents.py`, `util.py`, etc.) are from the [cshelton/pacman-ctf](https://github.com/cshelton/pacman-ctf) repository.


## Instructions

To use the `agent_eval.py` first put your agent under the `agents` directory. The run the script:
```
python3 agent_eval.py -a YOUR_TEAM_NAME -e ENEMY_TEAM_NAME  -n NUM_OF_EVALS
```
The result will be saved under the `results` directory.
The default values:

`-a`: `baselineTeam` 

`-e`: `baselineTeam`

`-n`: `100`

For QapproxTeam to be able to run, the PolicyFolder should be in the same folder with capture.py.
Then just run e.g 
```
python3 capture.py -r QapproxTeam -b baselineteam
```

### Train QapproxTeam

1. Place `PolicyFolder/` (containing `DefensiveQ.txt`, `OffensiveQ.txt`, `savebufferDefensiveQ.txt`, `savebufferOffensiveQ.txt`) in the same folder as `capture.py`
2. In `QapproxTeam.py`, comment line 27 and uncomment line 28 to enable training mode
3. Run:
```bash
python3 capture.py -r QapproxTeam -b baselineTeam -n 1000 -Q
```
After training, revert lines 27/28 to run the agent normally.
