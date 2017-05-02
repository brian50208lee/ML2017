import sys, random
from mytool import parser
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator


test_file = sys.argv[0] if len(sys.argv) > 0 else './data/test.csv'
predict_file = sys.argv[1] if len(sys.argv) > 1 else './predict.csv'

# load data
test_X, test_Y = parser.parse(test_file)
test_X = test_X/255

def output():
	test_Y = model.predict(test_X)
	test_Y = np.argmax(test_Y,axis=1)
	output = open(predict_file,'w')
	output.write("id,label\n")
	for idx in xrange(len(test_Y)):
		output.write(str(idx) + "," + str(test_Y[idx]) + "\n")
	output.close()

model = load_model('./model_best/063834.h5')
output()

