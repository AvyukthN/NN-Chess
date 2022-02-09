import pandas as pd

df = pd.read_csv('./data/stockfish_eval-depth22/raw/chessData.csv')

evals = list(df['Evaluation'])

for i in range(len(evals)):
    if evals[i][0] == '#':
        evals[i] = int(evals[i][1:])
    if evals[i] == '\ufeff+23':
        evals[i] = 23
    else:
        evals[i] = int(evals[i])

evals.sort()

print(f'max - {evals[-1]} min - {evals[0]}')