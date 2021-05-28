import tensorflow as tf
from collections import OrderedDict
from operator import itemgetter 
import chess
import numpy as np
import pandas as pd

path_to_model = 'ChessProject/Model/linear_estimator/Model/1620506978'
path_to_transformed_model = 'ChessProject/Model/dnn_estimator/NN_Model/1621674271'


model = tf.saved_model.load(path_to_model)
transformed_model = tf.saved_model.load(path_to_transformed_model)

def predict(df_eval, imported_model = model):
    """Return array of predictions for each row of df_eval
    
    Keyword arguments:
    df_eval -- pd.DataFrame
    imported_model -- tf.saved_model 
    """
    col_names = df_eval.columns
    dtypes = df_eval.dtypes
    predictions = []
    for row in df_eval.iterrows():
      example = tf.train.Example()
      for i in range(len(col_names)):
        dtype = dtypes[i]
        col_name = col_names[i]
        value = row[1][col_name]
        if dtype == 'object':
          value = bytes(value, 'utf-8')
          example.features.feature[col_name].bytes_list.value.extend([value])
        elif dtype == 'float':
          example.features.feature[col_name].float_list.value.extend([value])
        elif dtype == 'int':
          example.features.feature[col_name].int64_list.value.extend([value])
      predictions.append(imported_model.signatures['predict'](examples = tf.constant([example.SerializeToString()])))
    return predictions

# Nhận vào một board và trả về array features của board như trong bộ dữ liệu 
def get_board_features(board):
    """Return array of features for a board
    
    Keyword arguments:
    board -- chess.Board()
    """
    board_features = []
    for square in chess.SQUARES:
      board_features.append(str(board.piece_at(square)))
    return board_features

# nhận vào một move, trả về 2 arrays from và to như trong bộ dữ liệu 
def get_move_features(move):
    """Return 2 arrays of features for a move
    
    Keyword arguments:
    move -- chess.Move
    """
    from_ = np.zeros(64)
    to_ = np.zeros(64)
    from_[move.from_square] = 1
    to_[move.to_square] = 1
    return from_, to_



# =============================================== predict dữ liệu đã biến đổi =================================================

def get_transformed_board_features(current_board):
  board_features = []
  for square in chess.SQUARES:
    piece = str(current_board.piece_at(square))
    if(piece == 'None'):
      board_features.append(0.0)
    elif(piece == 'P'):
      board_features.append( 1.0)
    elif(piece == 'p'):
      board_features.append(-1.0)
    elif(piece == 'N'):
      board_features.append( 3.0)
    elif(piece == 'n'):
      board_features.append(-3.0)
    elif(piece == 'B'):
      board_features.append( 3.5)  
    elif(piece == 'b'):
      board_features.append(-3.5)
    elif(piece == 'R'):
      board_features.append( 5.0)
    elif(piece == 'r'):
      board_features.append(-5.0)
    elif(piece == 'Q'):
      board_features.append( 9.0)
    elif(piece == 'q'):
      board_features.append(-9.0)
    elif(piece == 'K'):
      board_features.append( 10.0)
    elif(piece == 'k'):
      board_features.append(-10.0)
  return board_features

def get_transformed_move_features(move):
  from_ = []
  to_ = []
  from_.append(float(move.from_square))
  to_.append(float(move.to_square))
  return from_, to_

def predict_transformed(df_eval, imported_model = transformed_model):
    col_names = df_eval.columns
    dtypes = df_eval.dtypes
    predictions = []
    for row in df_eval.iterrows():
      example = tf.train.Example()
      for i in range(len(col_names)):
        dtype = dtypes[i]
        col_name = col_names[i]
        value = row[1][col_name]
        if dtype == 'object':
          value = bytes(value, 'utf-8')
          example.features.feature[col_name].bytes_list.value.extend([value])
        elif dtype == 'float':
          example.features.feature[col_name].float_list.value.extend([value])
        elif dtype == 'int':
          example.features.feature[col_name].int64_list.value.extend([value])
      predictions.append(imported_model.signatures['predict'](examples = tf.constant([example.SerializeToString()])))
    return predictions


#=============================================================================================================================

# Nhận vào current board và trả về panda's dataframe sử dụng cho quá trình dự đoán 
def get_possible_moves_data(current_board):
    """Return pd.DataFrame of all possible moves used for predictions
    
    Keyword arguments:
    current_board -- chess.Board()
    """
    data = []
    moves = list(current_board.legal_moves)
    for move in moves:
      '''
       lấy dữ liệu từ bộ dữ liệu chưa qua biến đổi 
      '''

      #from_square, to_square = get_move_features(move)
      #row = np.concatenate((get_board_features(current_board), from_square, to_square))
      '''
       Lấy dữ liệu từ bộ dữ liệu đã qua biến đổi 
      '''
      from_square, to_square = get_transformed_move_features(move)
      row = np.concatenate((get_transformed_board_features(current_board), from_square, to_square))
      data.append(row)
    
    board_feature_names = chess.SQUARE_NAMES
    #move_from_feature_names = ['from_' + square for square in chess.SQUARE_NAMES]
    #move_to_feature_names = ['to_' + square for square in chess.SQUARE_NAMES]
    move_from_feature_names = ['0']
    move_to_feature_names = ['1']

    columns = board_feature_names + move_from_feature_names + move_to_feature_names
    
    df = pd.DataFrame(data = data, columns = columns)

    for column in move_from_feature_names:
      df[column] = df[column].astype(float)
    for column in move_to_feature_names:
      df[column] = df[column].astype(float)
    return df

# Dự đoán các nước đi có tỉ lệ thắng lớn hơn proportion 
def find_best_moves(current_board, proportion = 0.5):
    """Return array of the best chess.Move
    
    Keyword arguments:
    current_board -- chess.Board()
    model -- tf.saved_model
    proportion -- proportion of best moves returned
    """
    moves = list(current_board.legal_moves)
    df_eval = get_possible_moves_data(current_board)
    #predictions = predict(df_eval)
    predictions = predict_transformed(df_eval)
    good_move_probas = []
    
    for prediction in predictions:
      proto_tensor = tf.make_tensor_proto(prediction['probabilities'])
      proba = tf.make_ndarray(proto_tensor)[0][1]
      good_move_probas.append(proba)
    
    dict_ = dict(zip(moves, good_move_probas))
    dict_ = OrderedDict(sorted(dict_.items(), key = itemgetter(1), reverse = True))

    best_moves = list(dict_.keys())

    return best_moves[0:int(len(best_moves)*proportion)]



'''
board = chess.Board()
print(board)
lst = find_best_moves(board, 0.75)
print(lst)
'''