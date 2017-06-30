## Python Version: 3.6

## Requuirement Python Libraries:
	xgboost
	pandas
	numpy
	sklearn
	scipy

## Usage:
	username$ cd src/GBDT
	1. Run by script:
		username$ bash xgb_best.sh
			or
		username$ bash xgb_best.sh <trainingData> <trainingLabel> <testData> <prediction_file> <submission_format>
	2. Run by python:
		username$ python xgb.py
			or
		username$ python xgb.py <trainingData> <trainingLabel> <testData> <prediction_file> <submission_format>

## Run Training: (this will auto save model to best_model folder after training)
	1. open xgb.py
	2. load_best_model = False