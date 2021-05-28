import glob
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

path_fischer = "ChessProject/Dataset/CSV_FISCHER"
path_morphy = "ChessProject/Dataset/CSV_MORPHY"
path_capablanca = "ChessProject/Dataset/CSV_CAPABLANCA" 

# glob: thư viện cho phép đọc các file có tên giống nhau 

files_fischer = glob.glob(path_fischer + "/*.csv")
files_morphy = glob.glob(path_morphy + "/*.csv")
files_capablanca = glob.glob(path_capablanca + "/*.csv")

li = []

for filename in files_fischer:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

train = pd.concat(li, axis=0, ignore_index=True)

train = shuffle(train)

print(train.shape)

# print(train.head())

features = list(train.iloc[:, 0:192].columns)

X = train[features]
Y = train['good_move']

# X, Y: Panda's dataframes
'''
    Mô tả bộ dữ liệu của các tuyển thủ thế giới: 
        830376 (mỗi example là một nước đi)
        193 cột:
            64 cột đầu (0-63): mô tả trạng thái bàn cờ hiện tại 
            64 cột tiếp theo (64-127): cho biết nước đi bắt đầu từ ô nào (1/0)
            64 cột tiếp theo (128-192): cho biết nước đi kết thúc ở ô nào  (1/0)
            Cột cuối cùng: Cho biết nước đi đó có phải nước đi tốt hay không 

    Nước đi tốt: nước đi mà người đi nước đó chiến thắng chung cuộc của trò chơi 
    
'''
categorical_columns = list(X.iloc[:, 0:63].columns)
numerical_columns = list(X.iloc[:, 64:192].columns)
feature_columns = []

# Biến đổi 
for feature_name in categorical_columns:
    vocabulary = X[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numerical_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype = tf.float32))

def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
          ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

# Chia data thành các batch 
def split_into_batches(df, batch_size=100000):
    nb_rows = len(df.index)
    intervals = []
  
    for i in range(0, nb_rows + 1, batch_size):
        intervals.append(i)
  
    if(intervals[-1] != nb_rows):
        intervals.append(nb_rows)
  
    batches_X = []
    batches_y = []
  
    for i in range(0, len(intervals) - 1):
        batches_X.append(train.iloc[intervals[i]:intervals[i + 1], :][features])
        batches_y.append(train.iloc[intervals[i]:intervals[i + 1], :]['good_move'])

    return batches_X, batches_y

# X_train, y_train là các tuple của các data_frame 
X_train, y_train = split_into_batches(train)

# model

# default linear classifier 
linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns, model_dir='ChessProject/Model')

input_functions = []
for df_X, df_y in zip(X_train, y_train):
    input_functions.append(make_input_fn(df_X, df_y))

i = 1
for input_function in input_functions:
    print('<======================================== NEW BATCH ========================================>')
    print('Batch: ' + str(i))
    i = i + 1
    linear_est.train(input_function)
  