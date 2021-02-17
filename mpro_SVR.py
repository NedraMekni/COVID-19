import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel,RFE,RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR,LinearSVR
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn import metrics
from numpy import transpose
from collections.abc import Iterable


fname='./PostEra_392_no_dipl.csv'



def imputing_missing_values(data):
	return SimpleImputer(missing_values=np.NaN, strategy='mean')


def write_ft(titles,features):
	print(titles)
	print(len(titles))
	with open('ft_selected.txt','w') as f:
		for ft in features:
			f.write(titles[ft]+'\n')


		
'''
target:   int, index of classification result
excluded: list, indexes to be excluded for each line
'''


def read_data(fname,target,excluded=[]):
	classification,titles,data = [],[],[]
	
	with open (fname, 'r') as ds:
		titles=ds.readline().strip().split(';')
		target=len(titles)+target if target<0 else target
		titles=[titles[i] for i in range(len(titles)) if i not in excluded and i != target]
		for x in ds:
			line=x.strip().split(';')
			classification.append(float(line[target]))
			data.append([float(line[i]) if len(line[i])>0 else np.NaN for i in range(len(line)) if i not in excluded and i != target ]) 
	return np.array(data),np.array(classification),titles

def constant_columns(X):
	X = X.transpose()
	col_0s = [i1  for i1 in range(len(X)) if all([X[i1][i]==X[i1][0] for i in range(len(X[0]))])]
	print('Deleting columns {}'.format(col_0s))
	print(len(col_0s))
	return col_0s


'''
Given dataset and target as numpy array returns the n main features,
where:
	X: dataset
	y: classification results
	n: number of feature in output

The target values y (class labels in classification, real numbers in regression).


'''

def feature_selection_RFECV_reg(X,y,n_tree):
	sel = RandomForestRegressor(n_estimators=n_tree,random_state=0)
	rfecv = RFECV(estimator=sel, step=1, scoring='r2')

	selector = rfecv.fit(X,y)
	print(selector.support_)
	print(selector.ranking_)

	print("Optimal number of features : %d" % selector.n_features_)
	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
	plt.show()
	return [i for i in range(len(selector.support_)) if selector.support_[i]==True]



def show_heatmap(X_train):
	correlation_mat_clean=X_train.corr().abs()
	upper_half=correlation_mat_clean.where(np.triu(np.ones(correlation_mat_clean.shape), k=1).astype(np.bool))
	plt.figure(figsize=(50, 50))
	upper_filt_hmap=sns.heatmap(upper_half)
	plt.show()


def log(X, s):
	print('#'*10+'\nlenght of {}: {}\n length of one line of {}: {}\n{}\n'.format(s,len(X),s,1 if not isinstance(X[0], Iterable) else len(X[0]),X)+'#'*10)


def dump_model(fname,model,X_test,y_test):
	joblib.dump(model, fname)




if __name__ == '__main__':
	
	if len(sys.argv)!=2:
		print('Usage: python3 prog <n_tree>')
		exit(1)	
	n_tree = int(sys.argv[1])

	print('INPUT: n_tree {}'.format(n_tree))


	target, excluded = -1,[0] # target = cluster col, excluded = list of columns to exclude

	X,y,titles = read_data(fname,target,excluded)	# titles do not consider target and excluded
	#log(X,'X after reading file')
	#log(y,'y after reading file')
	#log(titles,'titles after reading file')
	
	'''
	Normalize before feature selection seems to be a best practise -> https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization
	
	For tiny datasets substitute null values with sample mean seems to be a valid choice -> https://www.datasciencelearner.com/deal-with-missing-data-python/

	need check on better scientific sources
	'''
	random_state = np.random.RandomState(42)
	X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=random_state)

	log(X_train,'X_train after train_test_split')
	log(X_test,'X_train after train_test_split')
	log(y_train,'X_train after train_test_split')
	log(y_test,'X_train after train_test_split')


	# managing NULL values
	
	imp = imputing_missing_values(X_train)
	imp.fit(X_train)
	X_train = imp.transform(X_train)	
	X_test = imp.transform(X_test)

	# managing constant columns checking on train set and remove them from train and test set

	col_0s = constant_columns(X_train)
	X_train = [[X_train[i][i1] for i1 in range(len(X_train[0])) if i1 not in col_0s] for i in range(len(X_train))]
	X_test = [[X_test[i][i1] for i1 in range(len(X_test[0])) if i1 not in col_0s] for i in range(len(X_test))]
	titles = [titles[i] for i in range(len(titles)) if i not in col_0s]
	
	
	# Correlation matrix	
	df=pd.DataFrame(X_train, columns=titles)

	print(df)

	correlation_mat=df.corr().abs()
	print(correlation_mat.shape)
	upper_half=correlation_mat.where(np.triu(np.ones(correlation_mat.shape), k=1).astype(np.bool))
	
	# High correlated descriptors to remove
	to_drop=[column for column in upper_half.columns if any(upper_half[column]>0.60)]
	
	#show_heatmap(df)
	X_train_clean=df.drop(df[to_drop], axis=1)

	# Write csv
	#X_train_clean.to_csv('dataset_cleaned.csv')

	# Show heatmap 	
	#show_heatmap(X_train_clean)

	log(to_drop,'high correlated features')


	# Remove to_drop features from train, test and titles
	X_train = X_train_clean.to_numpy()
	X_test = [[X_test[i][i1] for i1 in range(len(X_test[0])) if titles[i1] not in to_drop] for i in range(len(X_test))]
	titles = [titles[i] for i in range(len(titles)) if titles[i] not in to_drop]


	log(X_train,'X_train after remove high correlated ft')
	log(X_test,'X_test after remove high correlated ft')
	log(titles,'titles after remove high correlated ft')

	# Standard Scaler computed from train set and applied to train and test

	X_train=np.array(X_train)
	X_test=np.array(X_test)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	# test is normalized according to train scaler
	X_test = scaler.transform(X_test)

	

	#log(X_train,'X_train before feature selection')
	#log(y_train,'y_train before feature selection')

	ft_indices = feature_selection_RFECV_reg(X_train,y_train,n_tree)
	print([titles[i] for i in ft_indices])
	write_ft(titles,ft_indices)
	X_train = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X_train]
	X_test = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X_test]
	X = [[el[i] for i in range(len(el)) if i in ft_indices] for el in X]
	
	log(X_train,'X_train after feature selection')
	log(ft_indices,'ft_indices feature selected')
	
	sel_descr=[titles[i] for i in ft_indices]
	
	parameters = {'kernel':['rbf'], \
	'C':[eval('1e+{}*0.001'.format(x)) for x in range(6)],\
	'gamma': [eval('1e-{}'.format(x)) for x in range(4)],\
	'epsilon': [eval('1e+{}*0.001'.format(x)) for x in range(6)]}
	


	svr=svm.SVR()
	print('GridSearchCV\n'+'-'*12)
	clf = GridSearchCV(svr, parameters)
	clf.fit(X_train,y_train)
	f=clf.best_params_
	print('best parameters ', f)
	print('-'*12)


	confidence=clf.score(X_test,y_test)

	y_pred=clf.predict(X_test)

	print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
	print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	print('R square: ', confidence)

	# saving model

	dump_model('Reregressor.sav',clf,X_test,y_test)

 
	
