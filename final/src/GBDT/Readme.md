## Python Version: 3.6 <br>

## Requuirement Python Libraries: <br>
	xgboost <br>
	pandas <br>
	numpy <br>
	sklearn <br>
	scipy <br>

## Usage: <br>
	username$ cd src/GBDT <br>
	1. Run by script: <br>
		username$ bash xgb_best.sh <br>
			or <br>
		username$ bash xgb_best.sh <trainingData> <trainingLabel> <testData> <prediction_file> <submission_format> <br>
	2. Run by python: <br>
		username$ python xgb.py <br>
			or <br>
		username$ python xgb.py <trainingData> <trainingLabel> <testData> <prediction_file> <submission_format> <br>

## Run Training: (this will auto save model to best_model folder after training) <br>
	1. open xgb.py <br>
	2. load_best_model = False <br>