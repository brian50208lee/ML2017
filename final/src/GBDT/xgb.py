import sys
import xgboost
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
import datetime
	
submission_format_path = 'SubmissionFormat.csv'
train_value_path = sys.argv[1] if len(sys.argv) > 1 else 'TrainValue.csv'
train_label_path = sys.argv[2] if len(sys.argv) > 2 else 'TrainLabel.csv'
test_path = sys.argv[3] if len(sys.argv) > 3 else 'Test.csv'
prediction_path = sys.argv[4] if len(sys.argv) > 4 else 'prediction.csv'

load_best_model = True
best_model_path = 'best_model/xgb_best.bin'

def print_params(params):
	print('----- Parameters -----')
	for key, val in params.items():
		print('{} = {}'.format(key, val))
	print('----- Parameters -----')

# set parameters
params = 	{
				# XGBoost Parameters
				'seed': 0,
				'silent': 1,
				'nthread': 4,
				'eval_metric': 'mlogloss',
				'objective': 'multi:softmax',
				'num_class': 3,
				'max_depth': 64,
				'num_round': 1000,
				'eta': 0.1,
				'gamma': 0,
				'subsample': 0.82,
				'colsample_bytree': 0.5,
			}
print_params(params)

def date_parser(train, test):
	# concate to df
	train_index = train.index.values
	test_index = test.index.values
	df = pd.concat([train, test], axis=0)

	# parse date
	date_recorder = list(map(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'), df['date_recorded'].values))
	df['year_recorder'] = list(map(lambda x: int(x.strftime('%Y')), date_recorder))
	df['weekday_recorder'] = list(map(lambda x: int(x.strftime('%w')), date_recorder))
	df['yearly_week_recorder'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
	df['month_recorder'] = list(map(lambda x: int(x.strftime('%m')), date_recorder))
	df['age'] = df['year_recorder'].values - df['construction_year'].values
	del df['date_recorded']

	# factorize object
	for col in df.columns.values:
		if df[col].dtype.name == 'object':
			df[col] = df[col].factorize()[0]

	# split to trina and test
	return (df.loc[train_index], df.loc[test_index])

# read train data and test data
train = pd.DataFrame.from_csv(train_value_path)
test = pd.DataFrame.from_csv(test_path)
train, test = date_parser(train, test)

# read train labels
train_labels = pd.DataFrame.from_csv(train_label_path)
label_encoder = LabelEncoder()
train_labels.iloc[:, 0] = label_encoder.fit_transform(train_labels.values.flatten())

if load_best_model:
	# build final model
	xg_train = xgboost.DMatrix(train, label=train_labels.values.flatten())
	xg_test = xgboost.DMatrix(test)

	xgclassifier = xgboost.Booster(params)
	xgclassifier.load_model(best_model_path)
else:
	# find best boost round
	all_best_rounds = []
	kf = 	StratifiedKFold(
				train_labels.values.flatten(), 
				n_folds=4, 
				shuffle=True, 
				random_state=0
			)
	for cv_train_index, cv_test_index in kf:
		xg_train = xgboost.DMatrix(train.values[cv_train_index, :], label=train_labels.iloc[cv_train_index].values.flatten())
		xg_test = xgboost.DMatrix(train.values[cv_test_index, :], label=train_labels.iloc[cv_test_index].values.flatten())

		xgclassifier = 	xgboost.train(
							params, xg_train, 
							num_boost_round=params['num_round'], 
							evals=[(xg_train, 'train'), (xg_test, 'test')],
							early_stopping_rounds=50
						)
		all_best_rounds.append(xgclassifier.best_iteration)
	best_boost_round = int(np.mean(all_best_rounds))
	print('The best n_rounds is %d' % best_boost_round)

	# build final model
	xg_train = xgboost.DMatrix(train, label=train_labels.values.flatten())
	xg_test = xgboost.DMatrix(test)

	final_round = int(best_boost_round * 1.2)
	xgclassifier = xgboost.train(params, xg_train, final_round, evals=[(xg_train, 'train')])
	xgclassifier.save_model(best_model_path)

# prediction
print('writing to file')
preds = xgclassifier.predict(xg_test).astype(int)
preds = label_encoder.inverse_transform(preds)
submission_file = pd.DataFrame.from_csv(submission_format_path)
submission_file['status_group'] = preds
submission_file.to_csv(prediction_path)

