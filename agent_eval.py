import argparse
import pandas as pd
import os 
import subprocess
import time

parse = argparse.ArgumentParser()
parse.add_argument('--agent', '-a', type=str, default='baselineTeam')
parse.add_argument('--enemy', '-e', type=str, default='baselineTeam')
parse.add_argument('--n_rounds', '-n', type=int, default=100)

args = parse.parse_args()
print(args)
OUT_FILE = args.agent + '_vs_' + args.enemy + '.csv'
results = {'win': 0, 'draw': 0, 'lose': 0}
print('Start evaluation')
command = f"python3 capture.py -r {os.path.join('agents',args.agent) } -b {os.path.join('agents',args.enemy)}  -Q".split()
print(command)

# subprocess.run(command)

print("STARTING EVALUATION")
# print(result)
# print(result.stdout.decode())
# exit()
for _ in range(args.n_rounds):
    print(f'Round {_+1}/{args.n_rounds}')
    result = subprocess.run(command, stdout=subprocess.PIPE)
    with open("score", "r") as f:
        score = int(f.read())
    # print(score)
    results['win' if score > 0 else 'draw' if score == 0 else 'lose'] += 1

    # print(result.stdout.decode())

print('Evaluation done')
print(results)
df = pd.DataFrame(results, index=[0])
df.to_csv(os.path.join("results",OUT_FILE), index=False)