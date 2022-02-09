from flask import Flask, request, Response, redirect
import base64
import chess
import chess.svg
import random
from preprocess import piece_masks# , # flatten_masks
from net import NN
import torch

bd = chess.Board()

model = NN(input_size=769)
model.load_state_dict(torch.load('./models/model-769vec100epochs'))

app = Flask(__name__)

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def home():
    ret = '<html><body><img width=600 height=600 src="board.svg"></img></body></html>'
    ret += '</br>'
    ret += '</br>'
    ret += '</br>'
    ret += '<form action="/move"><input name="move" type="text" style="height:50px; width: 500px;"></input><input type="submit" value = "Move" style="height:50px; width: 70px;"></form?<br/>'

    return ret

@app.route('/board.svg', methods=['GET', 'POST'])
def play():
    return Response(chess.svg.board(board=bd), mimetype='image/svg+xml')

@app.route('/move', methods=['GET', 'POST'])
def move():
    move = request.args.get('move', default="")
    if move is not None and move != "":
        print(f'human move - {move}')
        # bd.Move(from_square=move[1:3], to_square=move[2:])
        bd.push_san(move)

        # INSERT NN MOVES HERE
        leg_moves = list(bd.legal_moves)

        best_move_rating = float('-inf')
        best_move = ""
        for move in leg_moves:
            move_str = str(move)

            bd.push_san(move_str)
            board_state = [temp.split(' ') for temp in str(bd).split('\n')]

            vec = piece_masks(board_state)
            vec.append(-1)

            move_rating = model(torch.Tensor(vec))
            bd.pop()

            if move_rating > best_move_rating:
                best_move = move_str
                best_move_rating = move_rating

        bd.push_san(best_move)

    print(move)

    return home()

if __name__ == '__main__':
    app.run()