import chess 
import numpy as np    
    # các mảng đánh giá điểm số của từng quân cờ theo vị trí 
pawn_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                                [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
                                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                                [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                                [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float)
    
pawn_black_eval = pawn_white_eval[::-1]

knight_white_eval = np.array([[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                                  [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
                                  [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
                                  [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
                                  [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
                                  [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
                                  [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
                                  [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]], np.float)
                            
knight_black_eval = knight_white_eval[::-1]

bishop_white_eval = np.array([[-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                                  [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                  [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
                                  [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
                                  [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
                                  [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                                  [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
                                  [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]], np.float)
    
bishop_black_eval = bishop_white_eval[::-1]

rook_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
                                [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [ 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]], np.float)

rook_black_eval = rook_white_eval[::-1]

    
queen_white_eval = np.array([[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                                 [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                 [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                                 [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                                 [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                                 [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                                 [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                                 [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]], np.float)
                        
queen_black_eval = queen_white_eval[::-1]

king_white_eval = np.array([[-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                                [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                                [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]], np.float)

king_black_eval = king_white_eval[::-1]

'''
    Python chess's board đánh số theo thứ tự từ 0->63
    Để có thể sử dụng mảng hai chiều mà ta đã tạo trước, cần chuyển 0->63 về tọa độ trên ô bàn cờ 
'''

def square_to_coor(square):
    return {
             0:(7,0),  1:(7,1),  2:(7,2),  3:(7,3),  4:(7,4),  5:(7,5),  6:(7,6),  7:(7,7),
             8:(6,0),  9:(6,1), 10:(6,2), 11:(6,3), 12:(6,4), 13:(6,5), 14:(6,6), 15:(6,7), 
            16:(5,0), 17:(5,1), 18:(5,2), 19:(5,3), 20:(5,4), 21:(5,5), 22:(5,6), 23:(5,7),
            24:(4,0), 25:(4,1), 26:(4,2), 27:(4,3), 28:(4,4), 29:(4,5), 30:(4,6), 31:(4,7),
            32:(3,0), 33:(3,1), 34:(3,2), 35:(3,3), 36:(3,4), 37:(3,5), 38:(3,6), 39:(3,7),
            40:(2,0), 41:(2,1), 42:(2,2), 43:(2,3), 44:(2,4), 45:(2,5), 46:(2,6), 47:(2,7),
            48:(1,0), 49:(1,1), 50:(1,2), 51:(1,3), 52:(1,4), 53:(1,5), 54:(1,6), 55:(1,7),
            56:(0,0), 57:(0,1), 58:(0,2), 59:(0,3), 60:(0,4), 61:(0,5), 62:(0,6), 63:(0,7)}[square]

def get_piece_value(piece, square, ai_white = True):
    if(ai_white):
        sign_white = 1.0
        sign_black = -1.0
    else:
        sign_white = -1.0
        sign_black = 1.0
        
    x,y = square_to_coor(square)
    if(piece == "None"):
        return 0.0
    elif(piece == 'P'):
        return sign_white * (10.0 + pawn_white_eval[x][y])
    elif(piece == 'N'):
        return sign_white * (30.0 + knight_white_eval[x][y])
    elif(piece == 'B'):
        return sign_white * (30.0 + bishop_white_eval[x][y])
    elif(piece == 'R'):
        return sign_white * (50.0 + rook_white_eval[x][y])
    elif(piece == 'Q'):
        return sign_white * (90.0 + queen_white_eval[x][y])
    elif(piece == 'K'):
        return sign_white * (900.0 + king_white_eval[x][y])
    elif(piece == 'p'):
        return sign_black * (10.0 + pawn_black_eval[x][y])
    elif(piece == 'n'):
        return sign_black * (30.0 + knight_black_eval[x][y])
    elif(piece == 'b'):
        return sign_black * (30.0 + bishop_black_eval[x][y])
    elif(piece == 'r'):
        return sign_black * (50.0 + rook_black_eval[x][y])
    elif(piece == 'q'):
        return sign_black * (90.0 + queen_black_eval[x][y])
    elif(piece == 'k'):
        return sign_black * (900.0 + king_black_eval[x][y])
    
def basic_eval(current_board, ai_white = True):
    evaluation = 0
    for square in chess.SQUARES:
        piece = str(current_board.piece_at(square))
        evaluation = evaluation + get_piece_value(piece, square, ai_white)
    return evaluation