import chess
import numpy as np
import evaluate
import predict 

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

# minimax algorithm without Alpha-Beta prunning 
def basic_minimax(current_board, depth, is_maximizing_player, ai_white = True):
    if(depth == 0):
        return - evaluate.basic_eval(current_board, ai_white)
    
    if(is_maximizing_player):
        best_move = -9999
        for move in list(current_board.legal_moves):
            current_board.push(move)
            curr = basic_minimax(current_board, depth-1, not is_maximizing_player, ai_white)
            best_move = max(best_move, curr)
            current_board.pop()
        return best_move

    else:
        best_move = 9999 
        for move in list(current_board.legal_moves):
            current_board.push(move)
            curr = basic_minimax(current_board, depth-1, not is_maximizing_player, ai_white)
            best_move = min(best_move, curr)
            current_board.pop()
        return best_move  


def minimax_alpha_beta(current_board, depth, alpha, beta, is_maximizing_player, ai_white = True):
    if(depth == 0):
        return - evaluate.basic_eval(current_board, ai_white)

    if(is_maximizing_player):
        best_move = - 9999
        for move in current_board.legal_moves:
            current_board.push(move)
            curr = minimax_alpha_beta(current_board, depth-1, alpha, beta, not is_maximizing_player, ai_white)
            best_move = max(best_move, curr)
            current_board.pop()
            alpha = max(alpha,best_move)
            if(beta <= alpha):
                return best_move
        return best_move

    else:
        best_move = 9999
        for move in current_board.legal_moves:
            current_board.push(move)
            curr = minimax_alpha_beta(current_board, depth-1, alpha, beta, not is_maximizing_player, ai_white)
            best_move = min(best_move, curr)
            current_board.pop()
            beta = min(beta, best_move)
            if(beta <= alpha):
                return best_move
        return best_move  


def minimax_with_ml(current_board, depth, alpha, beta, is_maximizing_player, ai_white = True):
    if(depth == 0):
        return - evaluate.basic_eval(current_board, ai_white)
    elif(depth > 3):
        legal_moves = predict.find_best_moves(current_board, 0.75)
    else:
        legal_moves = list(current_board.legal_moves)
    
    if(is_maximizing_player):
        best_move = -9999
        for move in legal_moves:
            current_board.push(move)
            best_move = max(best_move, minimax_with_ml(current_board, depth-1, alpha, beta, not is_maximizing_player, ai_white))
            current_board.pop()
            alpha = max(alpha, best_move)
            if(beta <= alpha):
                return best_move
        return best_move
    else:
        best_move = 9999
    for move in legal_moves:
        current_board.push(move)
        best_move = min(best_move, minimax_with_ml(current_board, depth-1, alpha, beta, not is_maximizing_player, ai_white))
        current_board.pop()
        beta = min(beta, best_move)
        if(beta <= alpha):
            return best_move
    return best_move
    

# ai turn to make a move
# ai is the maximizing player by default
def find_move(current_board, depth, is_maximizing_player = True, ai_white = True):
    best_move = -9999
    best_move_found = None
    for move in current_board.legal_moves:
        current_board.push(move)
        # minimax without alpha-beta prunning
        # value = basic_minimax(current_board, depth-1, not is_maximizing_player, ai_white)
        
        # minimax with alpha-beta (initially set at -10000 and 10000) prunning
        #value = minimax_alpha_beta(current_board, depth, -10000, 10000, is_maximizing_player, ai_white)

        # minimax with alpha_beta prunning and machine learning engine 
        value = minimax_with_ml(current_board, depth, -10000, 10000, is_maximizing_player, ai_white)

        if(value >= best_move):
            best_move = value
            best_move_found = move
        current_board.pop()
    return best_move_found
