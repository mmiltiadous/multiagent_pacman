# Multi-Agent Pacman Capture the Flag

## Folders/Files
- problem_description.pdf (Description of objectives)
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
