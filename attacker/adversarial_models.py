import copy
import os.path
import random

import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from copy import deepcopy

import shap
from argparser import parse
args = parse()
random.seed(args.seed)
np.random.seed(args.seed)

class Adversarial_Model(object):
	"""	A scikit-learn style adversarial explainer base class for adversarial models.  This accetps 
	a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
	data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
	Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.

	Parameters
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display):
		self.f_obscure = f_obscure
		self.psi_display = psi_display

		self.cols = None
		self.scaler = None
		self.numerical_cols = None

	def predict_proba(self, X, threshold=0.5):
		""" Scikit-learn style probability prediction for the adversarial model.  

		Parameters
		----------
		X : np.ndarray

		Returns
		----------
		A numpy array of the class probability predictions of the advesarial model.
		"""
		if self.perturbation_identifier is None:
			raise NameError("Model is not trained yet, can't perform predictions.")

		# generate the "true" predictions on the data using the "bad" model -- this is f in the paper
		predictions_to_obscure = self.f_obscure.predict_proba(X)

		# generate the "explain" predictions -- this is psi in the paper

		predictions_to_explain_by = self.psi_display.predict_proba(X)

		# in the case that we're only considering numerical columns
		if self.numerical_cols:
			X = X[:,self.numerical_cols]

		# allow thresholding for finetuned control over psi_display and f_obscure
		pred_probs = self.perturbation_identifier.predict_proba(X)
		perturbation_preds = (pred_probs[:,1] >= threshold)

		sol = np.where(np.array([perturbation_preds == 1,perturbation_preds==1]).transpose(), predictions_to_obscure, predictions_to_explain_by)

		return sol

	def predict(self, X):
		"""	Scikit-learn style prediction. Follows from predict_proba.

		Parameters
		----------
		X : np.ndarray
		
		Returns
		----------
		A numpy array containing the binary class predictions.
		"""
		pred_probs = self.predict_proba(X)
		return np.argmax(pred_probs,axis=1)

	def score(self, X_test, y_test):	
		""" Scikit-learn style accuracy scoring.

		Parameters:
		----------
		X_test : X_test
		y_test : y_test

		Returns:
		----------
		A scalar value of the accuracy score on the task.
		"""

		return np.sum(self.predict(X_test)==y_test) / y_test.size

	def get_column_names(self):
		""" Access column names."""

		if self.cols is None:
			raise NameError("Train model with pandas data frame to get column names.")

		return self.cols

	def fidelity(self, X):
		""" Get the fidelity of the adversarial model to the original predictions.  High fidelity means that
		we're predicting f along the in distribution data.
		
		Parameters:
		----------
		X : np.ndarray	

		Returns:
		----------
		The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
		"""

		return (np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0])

class Adversarial_Lime_Model(Adversarial_Model):
	""" Lime adversarial model.  Generates an adversarial model for LIME style explainers using the Adversarial Model
	base class.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	perturbation_std : float
	"""
	def __init__(self,  f_obscure, psi_display, perturbation_std=0.3):
		super(Adversarial_Lime_Model, self).__init__(f_obscure, psi_display)
		self.perturbation_std = perturbation_std



	def train(self, X, y, feature_names, perturbation_multiplier=30, categorical_features=[], rf_estimators=100, estimator=None):
		""" Trains the adversarial LIME model.  This method trains the perturbation detection classifier to detect instances
		that are either in the manifold or not if no estimator is provided.
		
		Parameters:
		----------
		X : np.ndarray of pd.DataFrame
		y : np.ndarray
		perturbation_multiplier : int
		cols : list
		categorical_columns : list
		rf_estimators : integer
		estimaor : func
		"""
		if isinstance(X, pd.DataFrame):
			cols = [c for c in X]
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names
		all_x, all_y = [], []

		# loop over perturbation data to create larger data set
		for _ in range(perturbation_multiplier):
			perturbed_xtrain = np.random.normal(0,self.perturbation_std,size=X.shape)
			p_train_x = np.vstack((X, X + perturbed_xtrain))
			p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

			all_x.append(p_train_x)
			all_y.append(p_train_y)


		all_x = np.vstack(all_x)
		all_y = np.concatenate(all_y)

		indices = np.arange(all_x.shape[0])
		np.random.shuffle(indices)
		all_x = all_x[indices,:]
		all_y = all_y[indices]







		# it's easier to just work with numerical columns, so focus on them for exploiting LIME
		self.numerical_cols = [feature_names.index(c) for c in feature_names if feature_names.index(c) not in categorical_features]
		self.numerical_cols = self.numerical_cols[:-1]

		# self.save_data(all_x,all_y,self.numerical_cols)

		if self.numerical_cols == []:
			raise NotImplementedError("We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model.")

		# generate perturbation detection model as RF
		xtrain = all_x[:,self.numerical_cols]
		xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain, ytrain)
		else:
			# self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)
			param_dist = {'n_estimators': 100,
						  'max_depth': 3,'random_state':10, 'seed':10}

			param_dist['objective'] = 'binary:logistic'

			self.perturbation_identifier = XGBClassifier(param_dist)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
	""" SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display):
		super(Adversarial_Kernel_SHAP_Model, self).__init__(f_obscure, psi_display)

	def train(self, X, y, feature_names, background_distribution=None, perturbation_multiplier=10, n_samples=2e4, rf_estimators=100, n_kmeans=10, estimator=None):
		""" Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from 
		its kmeans and randomly adding features.  These points get substituted into a test set.  We also check to make 
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is 
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		n_samples : int or float
		rf_estimators : int
		n_kmeans : int
		estimator : func

		Returns:
		----------
		The model itself.
		"""

		if isinstance(X, pd.DataFrame):
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names

		# This is the mock background distribution we'll pull from to create substitutions
		if background_distribution is None:
			background_distribution = shap.kmeans(X,n_kmeans).data
		repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

		new_instances = []
		equal = []

		# We generate n_samples number of substutions
		for _ in range(int(n_samples)):
			i = np.random.choice(X.shape[0])
			point = deepcopy(X[i, :])

			# iterate over points, sampling and updating
			for _ in range(X.shape[1]):
				j = np.random.choice(X.shape[1])
				point[j] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]),j])
	
			new_instances.append(point)

		substituted_training_data = np.vstack(new_instances)
		all_instances_x = np.vstack((repeated_X, substituted_training_data))

		# make sure feature truly is out of distribution before labeling it
		xlist = X.tolist()
		ys = np.array([1 if substituted_training_data[val,:].tolist() in xlist else 0\
						 for val in range(substituted_training_data.shape[0])])

		all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]),ys))

		xtrain,xtest,ytrain,ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain,ytrain)
		else:
			# self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain,ytrain)
			param_dist = {'n_estimators': 100,
						  'max_depth': 3, 'random_state': 10, 'seed': 10}

			param_dist['objective'] = 'binary:logistic'

			self.perturbation_identifier = XGBClassifier(**param_dist)

			self.perturbation_identifier.fit(xtrain, ytrain)
		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self


class Adversarial_Lime_Model1(Adversarial_Model):
	""" Lime adversarial model.  Generates an adversarial model for LIME style explainers using the Adversarial Model
	base class.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	perturbation_std : float
	"""

	def __init__(self, Xtrain_, Xtest_, path,  columns, ss, xmin, xmax,f_obscure, psi_display, perturbation_std=0.3):
		super(Adversarial_Lime_Model1, self).__init__(f_obscure, psi_display)
		self.perturbation_std = perturbation_std
		self.columns = columns
		self.ycol = "is_not_ood"
		self.path = path
		self.Xtrain_ = Xtrain_
		self.Xtest_ = Xtest_

		if os.path.exists(self.path) is False:
			os.mkdir(self.path)
		self.ss = ss
		self.xmin = xmin
		self.xmax = xmax
		self.fname = path.strip("/").split("/")[-1] + ".csv"


	def transform_data(self, X):
		X_ = copy.deepcopy(X)
		X_ = self.ss.inverse_transform(X_)
		X_ = np.round(X_, 0)
		X_ = np.clip(X_, self.xmin, self.xmax)
		return X_


	def load_data(self):
		fname = self.path + self.fname
		fname = fname.split(".csv")[0]
		df_train = pd.read_csv(fname + ".csv")
		df_test = pd.read_csv(fname + "_test.csv")
		cols = df_train.columns
		cols_test = df_test.columns
		Xtrain = df_train[cols[:-1]].values
		ytrain = df_train[cols[-1]].values


		Xtest = df_test[cols_test[:-1]].values
		ytest = df_test[cols_test[-1]].values
		print("loaded data successfully LIME")

		return Xtrain, ytrain, Xtest, ytest


	def save_data(self, X, y, ypred,Xtest,Ytest,ytest_pred,indices):
		# X = self.transform_data(X)
		# self.numerical_cols = self.numerical_cols[:2]

		X = np.round(X,2)
		Xtest = np.round(Xtest,2)

		y = y.reshape(-1, 1)
		Ytest = Ytest.reshape(-1,1)
		actual_cols = list(self.columns[indices])
		# X = X[:, self.numerical_cols]
		res = np.concatenate((X, y), axis=1)
		res_test = np.concatenate((Xtest, Ytest), axis=1)
		actual_cols.append(self.ycol)
		df = pd.DataFrame(res, columns=actual_cols)
		df_test = pd.DataFrame(res_test,columns=actual_cols)
		df[self.ycol]= df[self.ycol].astype(int)
		df_test[self.ycol]= df_test[self.ycol].astype(int)

		fname = self.path + self.fname

		fname_test = fname.split(".csv")[0] + "_test.csv"

		df.to_csv(fname, index=False)
		print("saved",fname)

		df_test.to_csv(fname_test, index=False)

		np.save(self.path + "ypred.npy",np.array(ypred))

		np.save(self.path + "ypred_test.npy",np.array(ytest_pred))
		print(fname)


	def train(self, X, y, feature_names, perturbation_multiplier=30, categorical_features=[], rf_estimators=100,
			  estimator=None,xgb_estimators=100,include_indices=None,load=True):
		""" Trains the adversarial LIME model.  This method trains the perturbation detection classifier to detect instances
		that are either in the manifold or not if no estimator is provided.

		Parameters:
		----------
		X : np.ndarray of pd.DataFrame
		y : np.ndarray
		perturbation_multiplier : int
		cols : list
		categorical_columns : list
		rf_estimators : integer
		estimaor : func
		"""
		xtrain =None
		xtest = None
		ytrain= None
		ytest = None
		# it's easier to just work with numerical columns, so focus on them for exploiting LIME
		self.numerical_cols = [feature_names.index(c) for c in feature_names if
							   feature_names.index(c) not in categorical_features]
		features = list(set(self.numerical_cols + include_indices))
		self.cols = feature_names
		if load==True:
			try:
				# assert(0)
				xtrain, ytrain, Xtest, Ytest = self.load_data()
			except:
				print("could not load data")
		if xtrain is None and load==False:
			if isinstance(X, pd.DataFrame):
				cols = [c for c in X]
				X = X.values
			elif not isinstance(X, np.ndarray):
				raise NameError(
					"X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

			if isinstance(X, pd.DataFrame):
				cols = [c for c in X]
				self.Xtest_ = self.Xtest_.values
			elif not isinstance(X, np.ndarray):
				raise NameError(
					"X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))


			all_x, all_y = [], []


			mask = np.ones(X.shape)
			n= X.shape[0]
			if include_indices is not None:
				for idx in include_indices:
					if idx not in self.numerical_cols:
						mask[:,idx]=np.zeros(n)
			# loop over perturbation data to create larger data set
			for _ in range(perturbation_multiplier):
				perturbed_xtrain = np.random.normal(0, self.perturbation_std, size=X.shape)
				modified_perturbations = perturbed_xtrain*mask
				p_train_x = np.vstack((X, X + modified_perturbations))
				p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

				all_x.append(p_train_x)
				all_y.append(p_train_y)
				# break

			all_x = np.vstack(all_x)
			all_y = np.concatenate(all_y)

			indices = np.arange(all_x.shape[0])
			np.random.shuffle(indices)
			all_x = all_x[indices, :]
			all_y = all_y[indices]


			if self.numerical_cols == []:
				raise NotImplementedError(
					"We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model.")

			# generate perturbation detection model as RF

			xtrain = all_x[:, features]

			xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)



		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain, ytrain)
		else:
			# self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)
			param_dist = {'n_estimators': xgb_estimators,
						  'max_depth': 3, 'random_state':10, 'seed':10}

			param_dist['objective'] = 'binary:logistic'

			self.perturbation_identifier = XGBClassifier(**param_dist)

			self.perturbation_identifier.fit(xtrain,ytrain)
			# self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

		ypred=None
		if xtest is not None:
			ypred = self.perturbation_identifier.predict(xtest)
			self.ood_training_task_ability = (ytest, ypred)

		Xtest = self.Xtest_[:, features]

		ytrain_pred = self.perturbation_identifier.predict(xtrain)

		ytest_pred = self.perturbation_identifier.predict(Xtest)

		Ytest = np.ones(Xtest.shape[0])

		print("percent",np.mean(ytrain==0))

		if load==False:
			self.save_data(xtrain, ytrain, ytrain_pred, Xtest,Ytest,ytest_pred,features)

		return self


class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
	""" SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""

	def __init__(self, Xtrain_, Xtest_, path,  columns,f_obscure, psi_display):
		super(Adversarial_Kernel_SHAP_Model, self).__init__(f_obscure, psi_display)
		self.columns = columns
		self.ycol = "is_not_ood"
		self.path = path
		self.Xtrain_ = Xtrain_
		self.Xtest_ = Xtest_

		if os.path.exists(self.path) is False:
			os.mkdir(self.path)

		self.fname = path.strip("/").split("/")[-1] + ".csv"

	def load_data(self):
		fname = self.path + self.fname
		fname = fname.split(".csv")[0]
		df_train = pd.read_csv(fname + ".csv")
		df_test = pd.read_csv(fname + "_test.csv")
		cols = df_train.columns
		cols_test = df_test.columns
		Xtrain = df_train[cols[:-1]].values
		ytrain = df_train[cols[-1]].values

		Xtest = df_test[cols_test[:-1]].values
		ytest = df_test[cols_test[:-1]].values
		print("loaded data successfully SHAP")
		return Xtrain, ytrain, Xtest, ytest

	def save_data(self, X, y, ypred, Xtest, Ytest, ytest_pred):
		# X = self.transform_data(X)
		# self.numerical_cols = self.numerical_cols[:2]

		X = np.round(X,2)
		Xtest = np.round(Xtest,2)


		y = y.reshape(-1, 1)
		actual_cols = list(self.columns)
		# X = X[:, self.numerical_cols]
		res = np.concatenate((X, y), axis=1)
		res_test = np.concatenate((Xtest, Ytest), axis=1)
		actual_cols.append(self.ycol)
		df = pd.DataFrame(res, columns=actual_cols)
		df_test = pd.DataFrame(res_test, columns=actual_cols)
		df[self.ycol] = df[self.ycol].astype(int)
		df_test[self.ycol] = df_test[self.ycol].astype(int)

		fname = self.path + self.fname

		fname_test = fname.split(".csv")[0] + "_test.csv"

		df.to_csv(fname, index=False)
		print("save",fname)

		df_test.to_csv(fname_test, index=False)

		np.save(self.path + "ypred.npy", np.array(ypred))

		np.save(self.path + "ypred_test.npy", np.array(ytest_pred))

	def train(self, X, y, feature_names, background_distribution=None, perturbation_multiplier=10, n_samples=2e4,
			  rf_estimators=100, n_kmeans=10, estimator=None,xgb_estimators=20,exclude_features=[],load=True):
		""" Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from
		its kmeans and randomly adding features.  These points get substituted into a test set.  We also check to make
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		n_samples : int or float
		rf_estimators : int
		n_kmeans : int
		estimator : func

		Returns:
		----------
		The model itself.
		"""
		xtrain=None
		xtest=None
		ytrain=None
		ytest=None
		self.cols = feature_names
		if load==True:
			try:
				xtrain, ytrain, Xtest, Ytest = self.load_data()
				print("loaded data successfully shap")
			except:
				pass

		if xtrain is None and load==False:

			if isinstance(X, pd.DataFrame):
				X = X.values
			elif not isinstance(X, np.ndarray):
				raise NameError(
					"X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

			if isinstance(X, pd.DataFrame):
				self.Xtest_ = self.Xtest_.values
			elif not isinstance(X, np.ndarray):
				raise NameError(
					"X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(self.Xtest_)))

			# This is the mock background distribution we'll pull from to create substitutions
			if background_distribution is None:
				background_distribution = shap.kmeans(X, n_kmeans).data
			repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

			new_instances = []
			equal = []

			indices_to_perturb=[]
			for idx in range(len(feature_names)):
				if ((exclude_features is None) or (feature_names[idx] not in exclude_features)):
					indices_to_perturb.append(idx)

			num_idc_to_pert = len(indices_to_perturb)


			# We generate n_samples number of substutions
			for _ in range(int(n_samples)):
				i = np.random.choice(X.shape[0])
				point = deepcopy(X[i, :])

				# iterate over points, sampling and updating
				for _ in range(num_idc_to_pert):
					j1 = np.random.choice(num_idc_to_pert)
					j = indices_to_perturb[j1]
					point[j] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]), j])

				new_instances.append(point)

			substituted_training_data = np.vstack(new_instances)
			all_instances_x = np.vstack((repeated_X, substituted_training_data))

			# make sure feature truly is out of distribution before labeling it
			xlist = X.tolist()
			ys = np.array([1 if substituted_training_data[val, :].tolist() in xlist else 0 \
						   for val in range(substituted_training_data.shape[0])])

			all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), ys))

			xtrain, xtest, ytrain, ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)


		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain, ytrain)
		else:
			# self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)
			param_dist = {'n_estimators': xgb_estimators,
						  'max_depth': 3, 'random_state': 10, 'seed': 10}

			param_dist['objective'] = 'binary:logistic'

			self.perturbation_identifier = XGBClassifier(**param_dist)

			self.perturbation_identifier.fit(xtrain, ytrain)
			print("train score",self.perturbation_identifier.score(xtrain,ytrain))

		print("percent",np.mean(ytrain==0))

		ypred=None
		if xtest is not None:
			ypred = self.perturbation_identifier.predict(xtest)
			self.ood_training_task_ability = (ytest, ypred)

		Xtest = self.Xtest_

		ytrain_pred = self.perturbation_identifier.predict(xtrain)

		ytest_pred = self.perturbation_identifier.predict(Xtest)

		Ytest = np.ones(Xtest.shape[0]).reshape(-1, 1)

		if load==False:
			self.save_data(xtrain, ytrain, ytrain_pred, Xtest, Ytest, ytest_pred)




		return self
