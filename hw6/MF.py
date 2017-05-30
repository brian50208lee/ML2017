import os, sys

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model, load_weights
from keras.layers import Dense, Dropout, Embedding, Reshape, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_train = 'train.csv'
data_test = 'test.csv'
data_users = 'users.csv'
data_movies = 'movies.csv'
data_directory = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

# user parameter
K_FACTORS = 120

# load data
df_train = pd.read_csv(data_directory + data_train)
df_test = pd.read_csv(data_directory + data_test)
df_users = pd.read_csv(data_directory + data_users, sep='::')
df_movies = pd.read_csv(data_directory + data_movies, sep='::')

# fix name mapping
df_movies.rename(columns={'movieID':'MovieID'}, inplace=True)

# concate train and test
train_UserID = df_train['UserID'].values
train_MovieID = df_train['MovieID'].values
train_Rating = df_train['Rating'].values.astype('float32')

test_DataID = df_test['TestDataID'].values
test_UserID = df_test['UserID'].values
test_MovieID = df_test['MovieID'].values

# max id
max_userid = np.max(train_UserID).astype('int')
max_movieid = np.max(train_MovieID).astype('int')

# shuffle
indices = np.arange(train_UserID.shape[0])  
np.random.shuffle(indices) 
train_UserID = train_UserID[indices]
train_MovieID = train_MovieID[indices]
train_Rating = train_Rating[indices]

# validation
valid_size = 5000
valid_UserID = train_UserID[:valid_size]
valid_MovieID = train_MovieID[:valid_size]
valid_Rating = train_Rating[:valid_size]
train_UserID  = train_UserID[valid_size:]
train_MovieID  = train_MovieID[valid_size:]
train_Rating  = train_Rating[valid_size:]

print('valid_UserID:', valid_UserID.shape)
print('valid_MovieID:', valid_MovieID.shape)
print('valid_Rating:', valid_Rating.shape)
print('train_UserID:', train_UserID.shape)
print('train_MovieID:', train_MovieID.shape)
print('train_Rating:', train_Rating.shape)


# build model
def build_model():
	model = Sequential()
	P = Sequential()
	P.add(Embedding(max_userid + 1, K_FACTORS, input_length=1))
	P.add(Reshape((K_FACTORS,)))
	Q = Sequential()
	Q.add(Embedding(max_movieid + 1, K_FACTORS, input_length=1))
	Q.add(Reshape((K_FACTORS,)))
	model.add(Merge([P, Q], mode='dot', dot_axes=1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
	model.summary()
	return model
model = build_model()

# train
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
checkpoint = ModelCheckpoint(
				filepath='./model/mf_weight_best.h5',
				verbose=1,
				save_best_only=True,
				save_weights_only=True,
				monitor='val_loss',
				mode='min'
			)
model.fit(
		[train_UserID, train_MovieID], train_Rating, 
		batch_size=1024, epochs=1, 
		validation_data=([valid_UserID,valid_MovieID],valid_Rating), 
		callbacks=[earlystopping, checkpoint]
	)

# predict
model = build_model()
model.load_weights('./model/mf_weight_best.h5')
y_preds = model.predict([test_UserID, test_MovieID]).reshape(-1)
df_pred = pd.DataFrame({'TestDataID': test_DataID, 'Rating': y_preds})
df_pred.to_csv(prediction_file, columns=['TestDataID','Rating'], index=False)





