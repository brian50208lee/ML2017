import sys, os
import numpy as np

from keras.preprocessing import text, sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


train_data_path = 'data' + os.sep + 'train_data.csv'
test_data_path = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test_data.csv'
output_file_path = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

TAG_INDEX_PATH = 'tag_index.npy'
WORD_INDEX_PATH = 'word_index.npy'
IDF_PATH = 'IDF.npy'

def load_train_data(data_path=train_data_path):
	print('Load training data...')
	ids, tags, texts = [], [], []
	for line in open(data_path, encoding='utf8').readlines()[1:]:
		_id, _tag, _text = line.strip().split(',',2)
		_tag = _tag.strip('"').split()
		ids.append(_id)
		tags.append(_tag)
		texts.append(_text)
	ids, tags, texts = np.array(ids), np.array(tags), np.array(texts)
	print('Texts shape:',texts.shape)
	print('Tags shape:',tags.shape)
	return (ids, tags, texts)

def load_test_data(data_path=test_data_path):
	print('Load testing data...')
	ids, texts = [], []
	for line in open(data_path, encoding='utf8').readlines()[1:]:
		_id, _text = line.strip().split(',',1)
		ids.append(_id)
		texts.append(_text)
	ids, texts = np.array(ids), np.array(texts)
	print('Texts shape:',texts.shape)
	return (ids, texts)

def save_dict(myDict, path):
	print('Saving dictionary to:', path)
	np.save(path, myDict)

def load_dict(path):
	print('Loading dictionary from:', path)
	myDict = np.load(path).item()
	return myDict

def tags_mapping(tags, tag_index=None):
	print ('Tags mapping...')

	if tag_index is None:
		tagset = list(set([t for tag in tags for t in tag]))
		tag_index = dict(zip(tagset, range(len(tagset)))) # create mapping dict	
	
	mTags = []
	for tag in tags:
		mTag = [0]*len(tag_index)
		for t in tag:
			mTag[tag_index[t]] = 1
		mTags.append(mTag)
	mTags = np.array(mTags)
	return mTags, tag_index

def texts_mapping(texts, word_index=None):
	print ('Tests mapping...')
	tokenizer = text.Tokenizer(filters='\'"’“”—‘´!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")
	if word_index is not None: # set word_index
		tokenizer.word_index = word_index
	else: # create word_index
		# concate text
		allTexts = texts.copy()
		if test_texts is not None:
			allTexts = np.concatenate([allTexts, test_texts])
		tokenizer.fit_on_texts(allTexts)

		# remove stopword
		'''
		stopword = ['the', 'and', 'of', 'a', 'to', 'in', 'is', 'his', 's', 'he', 'with', 'her', 'that', 'as', 'by', 'on', 'for', 'an', 'who', 'from', 'she', 'has', 'are', 'it', 'at', 'their', 'but', 'they', 'him', 'when', 'be', 'after', 'one', 'was', 'this', 'which', 'into', 'novel', 'life', 'not', 'out', 'have', 'new', 'up', 'all', 'them', 'two', 'about', 'world', 'been', 'also', 'will', 'old', 'only', 'where', 'first', 'other', 'while', 'there', 'find', 'war', 'can', 'or', 'had', 'more', 'home', 'years', 'being', 'himself', 'own', 'then', 'through', 'so', 'its', 'before', 'however', 'during', 'some', 'begins', 'between', 'what', 'becomes', 'end', 'must', 'now', 'over', 'takes', 'set', 'no',  'later', 'soon', 'way', 't', 'both', 'many', 'eventually', 'most', 'than', 'day', 'called', 'three', 'against', 'place', 'another', 'off', 'become', 'town', 'how', 'take', 'down', 'well', 'like', 'get', 'each', 'were', 'around', 'because', 'does', 'make', 'much', 'if']
		for k in stopword:
			try:
				del tokenizer.word_index[k]
			except:
				pass
		'''
	word_index = tokenizer.word_index # mapping dict
	sequences = tokenizer.texts_to_sequences(texts)
	return sequences, word_index

def texts_to_BOW_vectors(texts, word_index):
	print('Texts to BOW vector...')
	BOW_vectors = np.zeros((len(texts), (len(word_index)+1))).astype('float32')
	for idx, word_idxs in enumerate(texts):
		for word_idx in word_idxs:
			if word_idx <= len(word_index):
				BOW_vectors[idx][word_idx] += 1.0
	return BOW_vectors

def texts_TFIDF(texts, IDF=None):
	print('Texts TFIDF...')
	TF = texts.copy()
	TF = np.log(TF+1)

	if IDF is None:
		IDF = np.zeros(texts.shape)
		IDF[allTexts!=0] = 1.0
		IDF = IDF.sum(axis=0)
		IDF[IDF==0] = len(allTexts)
		IDF = np.log(len(allTexts)/IDF)
	TFIDF = TF*IDF
	return TFIDF, IDF

def f1_score(y_true,y_pred):
	thresh = 0.4
	y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	tp = K.sum(y_true * y_pred)

	precision=tp/(K.sum(y_pred))
	recall=tp/(K.sum(y_true))
	return 2*((precision*recall)/(precision+recall))

def build_BOW_model(word_index):
	print('Build BOW model...')
	model = Sequential()
	model.add(Dense(256, activation='relu', input_dim=len(word_index)+1))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(38, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[f1_score])
	model.summary()
	return model

def _save_model(path):
	model.save(path)

def _load_model(path):
	model = load_model(path, custom_objects={'f1_score':f1_score,})
	return model

def output(load_model_path):
	print('Output...')
	# load test data
	ids, test_texts = load_test_data()

	# load map
	tag_index = load_dict(TAG_INDEX_PATH)
	word_index = load_dict(WORD_INDEX_PATH)
	IDF = np.load(IDF_PATH)
	inverse_tag_index = dict(zip(tag_index.values(),tag_index))

	# pre-processing
	test_texts, word_index = texts_mapping(test_texts, word_index=word_index)
	test_texts = texts_to_BOW_vectors(test_texts, word_index)
	test_texts, IDF = texts_TFIDF(test_texts, IDF=IDF)

	# load model and predict
	model = _load_model(load_model_path)
	preds = model.predict(test_texts)

	# output
	out = open(output_file_path, 'w')
	out.write('"id","tags"\n')
	for _id, _pred in zip(ids, preds):
		line = '"{}","{}"\n'
		labels = ""
		for idx, val in enumerate(_pred):
			if val >= 0.5:
				labels += inverse_tag_index[idx] + " "
		if all(_pred < 0.5): # atleast 1 label
			max_id = np.argmax(_pred)
			labels += inverse_tag_index[max_id] + " "
		line = line.format(_id, labels.strip())
		out.write(line)
	out.close()

# load data
ids, train_tags, train_texts = load_train_data()
_, test_texts = load_test_data()

# concate
allTexts = np.concatenate([train_texts, test_texts])

# pre-processing
train_tags, tag_index = tags_mapping(train_tags)
allTexts, word_index = texts_mapping(allTexts)
allTexts =	texts_to_BOW_vectors(allTexts, word_index)
allTexts, IDF = texts_TFIDF(allTexts)

# save map
save_dict(tag_index, TAG_INDEX_PATH)
save_dict(word_index, WORD_INDEX_PATH)
np.save(IDF_PATH,IDF)

# de-concate
train_texts = allTexts[:len(train_texts)]
test_texts = allTexts[-len(test_texts):]

# validation
indices = np.arange(train_texts.shape[0])  
np.random.shuffle(indices) 
X, Y = train_texts[indices], train_tags[indices]
valid_size = 500
train_X, train_Y = X[valid_size:], Y[valid_size:]
valid_X, valid_Y = X[:valid_size], Y[:valid_size]

# train
model = build_BOW_model(word_index)
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 30, verbose=0, mode='max')
checkpoint = ModelCheckpoint(
				#filepath='./model/bow_f1_{val_f1_score:.3f}.h5',
				filepath='./model/bow_best.h5',
				verbose=1,
				save_best_only=True,
				save_weights_only=False,
				monitor='val_f1_score',
				mode='max'
			)
model.fit(train_X, train_Y,batch_size=128, epochs=10000, validation_data=(valid_X,valid_Y), callbacks=[earlystopping,checkpoint])

# output
output(load_model_path='model/bow_best.h5')