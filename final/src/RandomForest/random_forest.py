import util
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

def main():
	train()

def main():
	#f = open('result.txt', 'a')
	X, test = util.load_data()
	Y = util.load_target()
	#util.save_data(X, Y, test)
	#scaler = preprocessing.MinMaxScaler()
	#X = scaler.fit_transform(X)

	x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1)

	clf = RandomForestClassifier(n_estimators= 200)

	clf.fit(x_train, y_train)

	pred = clf.predict(x_val)

	print(accuracy_score(y_val, pred))

	pred =  clf.predict(test)

	util.predict(pred)

if __name__ == '__main__':
	main()