import sys, os
import numpy as np

from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras import backend as K

train_data = 'data' + os.sep + 'train_data.csv'
test_data = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test_data.csv'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

def load_data(data_path=train_data):
	print('Load data...')
	ids, tags, texts = [], [], []
	for line in open(data_path, encoding='utf8').readlines()[1:]:
		_id, _tag, _text = line.strip().split(',',2)
		_tag = _tag.strip('"').split()
		ids.append(_id)
		tags.append(_tag)
		texts.append(_text)
	ids, tags, texts = np.array(ids), np.array(tags), np.array(texts)
	return (ids, tags, texts)

def tags_mapping(tags, map_path='label_mapping.npy'):
	tagset = list(set([t for tag in tags for t in tag]))
	tagdict = dict(zip(tagset, range(len(tagset)))) # mapping dict
	mTags = []
	for tag in tags:
		mTag = [0]*len(tagdict)
		for t in tag:
			mTag[tagdict[t]] = 1
		mTags.append(mTag)
	return np.array(mTags)

def texts_mapping(texts, map_path='word_index.npy'):
	tokenizer = text.Tokenizer()
	tokenizer.fit_on_texts(texts)
	#print(tokenizer.word_index) # mapping dict
	return tokenizer.texts_to_sequences(texts)

def texts_padding(texts, maxlen=350):
	return sequence.pad_sequences(texts, maxlen=maxlen)

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall
def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision
def fbeta_score(y_true, y_pred, beta=1):
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
	    return 0
	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score

def build_model():
	print('Build model...')
	model = Sequential()
	model.add(Embedding(45969, 128))
	model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
	model.add(Dense(38, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[fbeta_score])
	model.summary()
	return model



if __name__ == '__main__':
	# load data
	ids, tags, texts = load_data()

	# pre-processing
	tags = tags_mapping(tags)
	texts = texts_mapping(texts)
	texts =	texts_padding(texts)

	# validation
	valid_size = 100
	train_X, train_Y = texts[:-valid_size][:], tags[:-valid_size][:]
	valid_X, valid_Y = texts[-valid_size:], tags[-valid_size:]

	# train
	model = build_model()
	model.fit(train_X, train_Y,batch_size=32, epochs=20, validation_data=(valid_X,valid_Y))

	# evaluate
	score, acc = model.evaluate(valid_X, valid_Y, batch_size=32)
	print('Test score:', score)
	print('Test accuracy:', acc)

