# Multi-Agent Pacman Capture the Flag
## Description

A multi-agent AI project built on the [Berkeley Pacman Capture the Flag framework](http://ai.berkeley.edu/contest.html) (Python 3 port by [cshelton](https://github.com/cshelton/pacman-ctf)). Two teams of agents compete to eat as many food pellets as possible from the opposing side while defending their own territory.

![Pacman CTF](capture_the_flag.png)
 
## Agents

Three agents were implemented, each using a different AI strategy:
1. QapproxTeam.py : Approximate Q-Learning (One offensive and one defensive agent that learn via function approximation Q-learning. Weights are trained over many episodes and saved to PolicyFolder/).
2. MCTSTeam.py : Monte Carlo Tree Search (Offensive agent uses MCTS when danger ghosts are nearby, falls back to a weighted heuristic otherwise. Defensive agent uses a heuristic policy).
3. HeuristicAgent.py : Rule-based Heuristic (Offensive and defensive agents guided by hand-crafted feature weights, like food distance, invader count, capsule distance, teammate distance, etc.).


## Folders/Files
- problem_description.pdf (Objectives)
- agents/ (Implemented agents)
- agent_eval.py (Script to evaluate agents over N games)
- PolicyFolder/ (Tuned Parameters)
- results/ (Evaluation results)
- report.pdf (Documentation of Implementation)


Check [pacman-ctf/](https://github.com/cshelton/pacman-ctf) for cloned files.
## Instructions

To use the `agent_eval.py` first put your agent under the `agents` directory. The run the script:
```
python3 agent_eval.py -a YOUR_TEAM_NAME -e ENEMY_TEAM_NAME  -n NUM_OF_EVALS
```
The result will be saved under the `results` directory.
The default values:

`-a`: `balineTeam` 

`-e`: `baselineTeam`

`-n`: `100`

For QapproxTeam to be able to run, the PolicyFolder should be in the same folder with capture.py.
Then just run e.g 
```
python3 capture.py -r QapproxTeam -b baselineteam
```

To train QapproxTeam and obtain the policy, a folder named policyFolder should be in the same folder with it and capture.py. The policyFolder should contain four .txt files with names: 

DefensiveQ, OffensiveQ, savebufferDefensiveQ, savebufferOffensiveQ.
Uncomment line 28 and comment line 27.

Then just run: 
```
python3 capture.py -r QapproxTeam -b baselineteam -n 1000 -Q
```
