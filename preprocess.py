import chess.pgn
import chess
import numpy as np
import pandas as pd

def piece_masks(board_state: list) -> list:
    P = list((np.array(board_state) == 'P').astype(int).flatten())
    p = list((np.array(board_state) == 'p').astype(int).flatten())
    B = list((np.array(board_state) == 'B').astype(int).flatten())
    b = list((np.array(board_state) == 'b').astype(int).flatten())
    N = list((np.array(board_state) == 'N').astype(int).flatten())
    n = list((np.array(board_state) == 'n').astype(int).flatten())
    R = list((np.array(board_state) == 'R').astype(int).flatten())
    r = list((np.array(board_state) == 'r').astype(int).flatten())
    Q = list((np.array(board_state) == 'Q').astype(int).flatten())
    q = list((np.array(board_state) == 'q').astype(int).flatten())
    K = list((np.array(board_state) == 'K').astype(int).flatten())
    k = list((np.array(board_state) == 'k').astype(int).flatten())

    return list(np.concatenate((P, B, N, R, Q, K, p, b, n, r, q, k), axis=0))

def parse_pgn(filepath: str, training_examples: int) -> tuple:
    X = []
    y = []
    
    result_encoding = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    threshold = training_examples // 2
    
    count_1 = 0
    
    for i in range(training_examples):
        pgn = chess.pgn.read_game(open(filepath))
            
        result = pgn.headers['Result']
        
        bd = chess.Board()

        if result_encoding[result] == 1:
            count_1 += 1
        
        if not(count_1 > threshold and result_encoding[result] == 1): 
            # print(result, i, count_1)
            white = True
            for i, move in enumerate(pgn.mainline_moves()):
                bd.push(move)
                board_state = [temp.split(' ') for temp in str(bd).split('\n')]

                input_vec = piece_masks(board_state)
                input_vec = list(map(str, input_vec))

                if white:
                    turn = '1'
                else:
                    turn = '-1'

                input_vec.append(turn)
                input_vec = ' '.join(input_vec)

                X.append(input_vec)
                y.append(result_encoding[result])
                # input_vec = [str(inp) for inp in input_vec]
                
                white = not(white)
        else:
            continue

    return X, y

if __name__ == '__main__':
    X, y = parse_pgn('./data/pgn_data/KingBase2018-02.pgn', 2000000)

    df = pd.DataFrame({
        'X': pd.Series(X),
        'y': pd.Series(y)
    })

    df.to_csv('./data/preprocessed/KingBasePGN.csv')

    # np.savez("./data/preprocessed/chess_data1.npz", X, y)
