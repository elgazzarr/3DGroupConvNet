import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def train():
	train_df = pd.read_csv('/data/agelgazzar/Work/AgePrediction/3DResnet/code/csvfiles/training_error.csv')
	X = train_df['error'].values + train_df['age'].values
	X_train = np.expand_dims(X,1)
	y = train_df['age']
	# Model initialization
	regression_model = LinearRegression()
	# Fit the data(train the model)
	regression_model.fit(X_train, y)
	# Predict
	y_predicted = regression_model.predict(X_train)

	# model evaluation
	rmse = mean_absolute_error(y, y_predicted)
	r2 = r2_score(y, y_predicted)

	# printing values
	print('Slope:' ,regression_model.coef_)
	print('Intercept:', regression_model.intercept_)
	print('Training  mean absoule error: ', rmse)
	print('Training R2 score: ', r2)
	print('-----------------------------------------------------')

	plt.scatter(X, y, s=10)
	plt.xlabel('Age')
	plt.ylabel('Predicted age')

	# predicted values
	plt.plot(X, y_predicted, color='r')
	plt.show()


	return regression_model

def test(model):
	train_df = pd.read_csv('/data/agelgazzar/Work/AgePrediction/3DResnet/code/csvfiles/test_error.csv')
	X = train_df['error'].values + train_df['age'].values
	x = np.expand_dims(X,1)		
	y = train_df['age']
	y_predicted = model.predict(x)
	rmse = mean_absolute_error(y, y_predicted)
	r2 = r2_score(y, y_predicted)
	print('Test mean absolute error: ', rmse)
	print('Test R2 score: ', r2)


	# model evaluation
	rmse = mean_absolute_error(y, y_predicted)
	r2 = r2_score(y, y_predicted)

	# printing values


	plt.scatter(X, y, s=10)
	plt.xlabel('Age')
	plt.ylabel('Predicted age')

	# predicted values
	plt.plot(X, y_predicted, color='r')
	plt.show()



if __name__ == '__main__':
	model = train()
	resutls = test(model)