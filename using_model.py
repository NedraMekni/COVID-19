
import joblib
import numpy as np
from sklearn.svm import SVR,LinearSVR


def read_data(fname,included=[]):
	titles,data = [],[]
	
	with open (fname, 'r') as ds:
		titles=ds.readline().strip().split(';')
		included = list(set([i for i in range(len(titles)) if titles[i] in included]))
		titles=[titles[i] for i in range(len(titles)) if i in included]
		for x in ds:
			line=x.strip().split(';')
			data.append([float(line[i]) if len(line[i])>0 else np.NaN for i in range(len(line)) if i in included]) 
	return np.array(data)

def read_sel_features(fname='ft_selected.txt'):
	features=[]
	with open (fname, 'r') as ds:
		for l in ds:
			features.append(l.strip())
	return features

def read_sel_features_scaler(fname='dimension_for_scaler.txt'):
	return eval(open(fname,'r').readline())

def filter_dataset(data,features_scaler,features_selected):
	features_selected =list(set([i for i in range(len(features_scaler)) if features_scaler[i] in features_selected]))
	return [[data[i][j] for j in range(len(data[0])) if j in features_selected] for i in range(len(data))]

if __name__ == '__main__':

	features_scaler=read_sel_features_scaler()
	features_selected = read_sel_features()
	data=read_data('PostEra_Non_cov_cy_MD.csv',features_scaler)

	scaler=joblib.load('std_scaler.bin')
	data = scaler.transform(data)	
	data=filter_dataset(data,features_scaler,features_selected)

	clf = joblib.load('Reregressor.sav')
	y_pred=clf.predict(data)

	print(y_pred)

