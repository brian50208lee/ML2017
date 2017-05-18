import sys, os
import numpy as np

from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras import backend as K

train_data = 'data' + os.sep + 'train_data.csv'
test_data = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test_data.csv'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

MAX_SEQUENCE_LENGTH = 350


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
	print('texts:',texts.shape)
	print('tags:',tags.shape)
	return (ids, tags, texts)

def tags_mapping(tags, map_path='label_mapping.npy'):
	print ('Tags mapping...')
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
	print ('Tests mapping...')
	tokenizer = text.Tokenizer()
	tokenizer.fit_on_texts(texts)
	word_index = tokenizer.word_index # mapping dict
	return tokenizer.texts_to_sequences(texts), word_index

def texts_padding(texts, maxlen=MAX_SEQUENCE_LENGTH):
	print('Tests padding...','maxlen:',maxlen)
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

def gloveEmbedding(word_index, embedding_dim=300, vector_path=os.sep.join(['data','glove.6B','glove.6B.{}d.txt'])):
	print('Load gloveEmbedding...','embedding_dim:',embedding_dim)
	# load all vectors from glove
	vector_path = vector_path.format(embedding_dim)
	word_vector_map = dict([(line.split()[0], np.array(line.split()[1:]).astype('float32')) for line in open(vector_path, encoding='utf8')])
	
	# create embadding matrix
	embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
	for word, idx in word_index.items():
		if word in word_vector_map:
			embedding_matrix[idx] = word_vector_map[word]

	# create embadding layer
	embeddingLayer = Embedding(
		len(word_index)+1, 
		embedding_dim, 
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False)

	return embeddingLayer

def build_model(word_index):
	print('Build model...')
	model = Sequential()
	embeddingLayer = gloveEmbedding(word_index)
	model.add(embeddingLayer)
	model.add(LSTM(embeddingLayer.output_dim, dropout=0.5, recurrent_dropout=0.5))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(38, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[fbeta_score])
	model.summary()
	return model



if __name__ == '__main__':

	# load data
	ids, tags, texts = load_data()

	# pre-processing
	tags = tags_mapping(tags)
	texts, word_index = texts_mapping(texts)
	texts =	texts_padding(texts)

	
	# validation
	valid_size = 100
	train_X, train_Y = texts[:-valid_size][:], tags[:-valid_size][:]
	valid_X, valid_Y = texts[-valid_size:], tags[-valid_size:]

	# train
	model = build_model(word_index)
	model.fit(train_X, train_Y,batch_size=32, epochs=30, validation_data=(valid_X,valid_Y))

	# evaluate
	score, acc = model.evaluate(valid_X, valid_Y, batch_size=32)
	print('Test score:', score)
	print('Test accuracy:', acc)
	
