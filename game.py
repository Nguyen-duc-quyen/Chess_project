import chess
import numpy as np
import os
import minimax

'''
    Notes:
        + Uppercase letters: white pieces
        + Lowercase letters: black pieces
        + "p" - pawn
        + "r" - rook
        + "n" - knight
        + "b" - bishop
        + "q" - queen
        + "k" - king 
        + White player ~ maximizing player 
        + Black player ~ minimizing Player
        + All the moves follow UCI standards
        

'''
def clear():
    os.system('clear')

def can_checkmate(move, current_board):
    fen = current_board.fen()
    future_board = chess.Board(fen)
    future_board.push(move)
    return future_board.is_checkmate()

def ai_turn(current_board, ai_white = True):
    clear()
    print(current_board)
    print("Number of possible moves: %s" %(len(list(current_board.legal_moves))))
    print("Wait a minute, I'm thinking")
    for move in current_board.legal_moves:
        if(can_checkmate(move, current_board)):
            current_board.push(move)
            return 
    
    number_of_moves = len(list(current_board.legal_moves))

    if(number_of_moves > 30):
        current_board.push(minimax.find_move(current_board, 1, ai_white))
    elif(number_of_moves > 10):
        current_board.push(minimax.find_move(current_board, 3, ai_white))
    else:
        current_board.push(minimax.find_move(current_board, 5, ai_white))
    return 

def human_turn(current_board):
    clear()
    print(current_board)
    move_uci = input('Enter your move: ')
    try: 
        move = chess.Move.from_uci(move_uci)
    except:
        return human_turn(current_board)
    if(move not in current_board.legal_moves):
        return human_turn(current_board)
    current_board.push(move)
    return 


def play_game(turn, current_board, ai_white = True):
    if(current_board.is_stalemate()):
        clear()
        print('Draw, both AI and human win')
        return
    else:
        if(not turn):
            if(not current_board.is_checkmate()):
                human_turn(current_board)
                return play_game(not turn, current_board, ai_white)
            else:
                clear()
                print(current_board)
                print('You lose, AI wins. Try again next time.')
                return
        else:
            if(not current_board.is_checkmate()):
                ai_turn(current_board, ai_white)
                return play_game(not turn, current_board, ai_white)
            else:
                clear()
                print(current_board)
                print('Congratulation, you win.')
                return


def game_start():
    ai_white = True
    board = chess.Board()
    print(board)
    human_white = input('Do you want to play as white? [y/ n]')
    if(human_white.lower() == 'y'):
        ai_white = False
        return play_game(False, board, ai_white)
    else:
        return play_game(True, board, ai_white)

game_start()
