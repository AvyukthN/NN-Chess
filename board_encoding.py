import pandas as pd
import numpy as np
from pprint import pprint
import chess
import time
import csv
from tqdm import tqdm

def fen_to_board(fen: str) -> list:
    board = []
    # sum = 0
    # min = float('inf')
    # max = float('-inf')
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend(['-'] * int(c))
            elif c == 'p':
                brow.append(c)
            elif c == 'P':
                brow.append(c)
            elif c > 'Z':
                brow.append(c)
            else:
                brow.append(c)

        board.append(brow)

    '''
    for i in range(len(board)):
        for j in range(len(board[i])):
            board[i][j] = ord(board[i][j])
            
            sum += board[i][j]
            if board[i][j] < min:
                min = board[i][j]
            if board[i][j] > max:
                max = board[i][j]
    '''

    return board

# segments the chess board into 6 different masks for each piece
# white -> 1
# black -> -1
# empty -> 0
def piece_masks(board: list) -> list:

    w_pawns = np.zeros((8, 8))
    w_bishops = np.zeros((8, 8))
    w_knights = np.zeros((8, 8))
    w_rooks = np.zeros((8, 8))
    w_queens = np.zeros((8, 8))
    w_kings = np.zeros((8, 8))

    b_pawns = np.zeros((8, 8))
    b_bishops = np.zeros((8, 8))
    b_knights = np.zeros((8, 8))
    b_rooks = np.zeros((8, 8))
    b_queens = np.zeros((8, 8))
    b_kings = np.zeros((8, 8))

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 'p':
                b_pawns[i][j] = 1
            elif board[i][j] == 'P':
                w_pawns[i][j] = 1
            elif board[i][j] == 'b':
                b_bishops[i][j] = 1
            elif board[i][j] == 'B':
                w_bishops[i][j] = 1
            elif board[i][j] == 'n':
                b_knights[i][j] = 1
            elif board[i][j] == 'N':
                w_knights[i][j] = 1
            elif board[i][j] == 'r':
                b_rooks[i][j] = 1
            elif board[i][j] == 'R':
                w_rooks[i][j] = 1
            elif board[i][j] == 'q':
                b_queens[i][j] = 1
            elif board[i][j] == 'Q':
                w_queens[i][j] = 1
            elif board[i][j] == 'k':
                b_kings[i][j] = 1
            elif board[i][j] == 'K':
                w_kings[i][j] = 1
    
    return w_pawns, w_bishops, w_knights, w_rooks, w_queens, w_kings, b_pawns, b_bishops, b_knights, b_rooks, b_queens, b_kings

def flatten_masks(masks: list, turn: int) -> np.ndarray:
    # input_vector = np.zeros((64*12), dtype=int)
    input_vector = np.zeros((64*12) + 1, dtype=int)
    input_vector[-1] = turn

    count = 0
    for i in range(len(masks)):
        temp_mask = masks[i].flatten()
        for j in range(len(temp_mask)):
            input_vector[count] = temp_mask[j]
            count += 1

    return input_vector

if __name__ == '__main__':
    # board = chess.Board()
    training_set = []

    data = pd.read_csv('./ANN_files/preprocessed_imp.csv')
    num_rows = len(data.index)

    # row = data.iloc[10000]

    # fen = '6k1/8/2p4p/P1p1q3/2N2N2/1P4K1/7r/8 w - - 0 50'
    # label = -13

    # # fen = board.fen().split(' ')
    # splitted = fen.split(' ')
    # board_fen = splitted[0]
    # turn = splitted[1]

    # if turn == 'w':
    #     turn = 1
    # else:
    #     turn = -1

    # masks = piece_masks(board_fen)
    # input_vec = list(flatten_masks(masks, turn))
    # input_vec = [str(inp) for inp in input_vec]

    # print(input_vec)
    # print(f'label - {label}')

    for i in tqdm(range(100000)):
        row = data.iloc[i]
        fen = row['FEN']
        print(fen)
        label = row['Evaluation']

        # fen = board.fen().split(' ')
        splitted = fen.split(' ')
        board_fen = splitted[0]
        turn = splitted[1]

        if turn == 'w':
            turn = 1
        else:
            turn = -1

        masks = piece_masks(fen_to_board(board_fen))
        input_vec = list(flatten_masks(masks, turn))
        input_vec = [str(inp) for inp in input_vec]

        training_set.append((' '.join(input_vec), label))

    df = pd.DataFrame(training_set)
    df.to_csv('preprocessed3_data.csv', index=False)