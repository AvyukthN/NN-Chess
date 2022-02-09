# CHESS with Neural Networks!!

## Basic Idea
> 1. Create a Artificial Neural Network to evaluate static chess boards
> 2. Use the static evaluation model to make decisions using Minimax with Alpha-Beta Pruning
> 3. Hope for the best

## Features
> ### States to Consider
> - board position
> - who's turn is it
> ### TO IMPLEMENT IN THE FUTURE
> - castling rights
> - 50-move rule (player can declare draw if no captures or pawn moves have been made in the last 50 moves)
> - move count (increased after each of black moves by 1)
> - en passant 

## Preprocessing Procedure
> ### Board State Preprocessing
> 1. Convert each fenstring into a 2d array of chess pieces
> 2. Segment the chess board into 6 masks corresponding to the 6 pieces
> 3. Each piece mask will be an 8x8 numpy array with values {1, -1, 0}
> - white piece -> 1 
> - black piece -> -1 
> - empty space -> 0
> 4. Flatten the masks, put them together and add a value for the turn at the end
>       - this gives us an input vector of shape (769, 1)
> ### Turn Preprocessing
> - input feature will be an 8x8 matrix of these values
> - 1 for white
> - -1 for black

## Neural Network
> ### Network Inputs
> - Input vector of combined flattened masks and turn -> (769, 1)

## Tree Search Algorithms
> ### Implementing
> - Minimax with Alpha Beta Pruning
>    - using the trained neural network as the static evaluation function
> ### Will Implement
> - Monte Carlo Tree Search

## To Play against the AI Easily
> ### Local Chess Server GUI
> - Will implement a simple SVG server in Flask to play chess with the trained AI