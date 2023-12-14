"""
The experiment MAIN for GERMAN.
"""
import os
import warnings

warnings.filterwarnings('ignore')
import joblib
from adversarial_models import *
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans

from copy import deepcopy

from argparser import parse
args = parse()
seed= args.seed
random.seed(args.seed)
np.random.seed(args.seed)


# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)



categorical_feature_names = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
               'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
               'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
               'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
               'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled','LoanRateAsPercentOfIncome']

not_categorical =[]
for f in cols:
    if f not in categorical_feature_names:
        not_categorical.append(f)

top_features = rank_top_features(X.values,y,cols,categorical_feature_names,not_categorical)
final_features = [cols[x] for x in top_features]
final_features.append('Gender')
final_features.append('LoanRateAsPercentOfIncome')
final_features = list(set(final_features))

print(final_features)

categorical_final = []
for feature in final_features:
    if feature in categorical_feature_names:
        categorical_final.append(feature)


X= X[final_features]

features = [c for c in X]



gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')
columns = X.columns

X = X.values
print("mean", np.mean(X[:5,:]))
print("std dev", np.std(X[:5,:]))
xmin = np.min(X, axis=0)
xmax = np.max(X, axis=0)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:, loan_rate_indc])

categorical = [features.index(c) for c in categorical_final]

print("categorical final",categorical_final)
print("final features", final_features)

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self, X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X):
        return one_hot_encode(np.array(
            [params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))


##
###
def save_results(path, results, filename=None):
    if os.path.exists(path)==False:
        os.mkdir(path)
    print("saved",path + filename + ".pkl")
    joblib.dump(results, path + filename + ".pkl")


def experiment_main(path, path_shap, ycol, columns, xmin, xmax,load=False,load_dataset=True):
    """
    Run through experiments for LIME/SHAP on GERMAN.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features
    """

    print('---------------------')
    print("Beginning LIME GERMAN Experiments....")
    print("(These take some time to run because we have to generate explanations for every point in the test set) ")
    print('---------------------')
    include_cols = [loan_rate_indc,gender_indc]
    # Train the adversarial model for LIME with f and psi
    adv_lime=None
    model_path = "../models/"
    if os.path.exists(model_path)==False:
        os.mkdir(model_path)

    if load==True:
        try:
            adv_lime = joblib.load(model_path + str(seed) +"_german_lmodified_adv_lime.pkl")
        except:
            pass
    if adv_lime is None:

        adv_lime = Adversarial_Lime_Model1(xtrain, xtest, path, columns, ss, xmin, xmax, racist_model_f(),
                                       innocuous_model_psi()).train(xtrain, ytrain, feature_names=features,
                                                                    perturbation_multiplier=30,
                                                                    categorical_features=categorical, xgb_estimators=50,include_indices=include_cols,load=load_dataset)
        joblib.dump(adv_lime,model_path + str(seed) +"_german_lmodified_adv_lime.pkl")

    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(),
                                                           discretize_continuous=False,
                                                           categorical_features=categorical)

    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

    # Display Results
    print("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
    e = experiment_summary(explanations, features)
    save_results(path, e, 'lime')

    print(e)
    print("Fidelity:", round(adv_lime.fidelity(xtest), 2))

    print('---------------------')
    print('Beginning SHAP GERMAN Experiments....')
    print('---------------------')

    # Setup SHAP
    exclude_features = []
    for feature in features:
        if feature in categorical_feature_names:
            exclude_features.append(feature)

    adv_shap = None
    background_distribution = None
    model_path = "../models/"
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)

    if load==True:
        try:
            adv_shap = joblib.load(model_path + str(seed) +"_german_lmodified_adv_shap.pkl")
            background_distribution = joblib.load(model_path + str(seed) + "_german_lmodified_adv_shap_bg.pkl")

        except:
            pass
    if adv_shap is None:
        adv_shap = Adversarial_Kernel_SHAP_Model(xtrain, xtest, path_shap, columns, racist_model_f(),
                                             innocuous_model_psi()).train(xtrain, ytrain,
                                                                          feature_names=features,
                                                                          background_distribution=background_distribution,
                                                                          rf_estimators=100, xgb_estimators=50,
                                                                             n_samples=5e4, exclude_features=exclude_features,load=load_dataset)
        joblib.dump(adv_shap,model_path + str(seed) +"_german_lmodified_adv_shap.pkl")
    if background_distribution is None:
        background_distribution = KMeans(n_clusters=10, random_state=0).fit(xtrain).cluster_centers_
        joblib.dump(background_distribution,model_path + str(seed) +"_german_lmodified_adv_shap_bg.pkl")

    adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
    explanations = adv_kerenel_explainer.shap_values(xtest)

    # format for display
    formatted_explanations = []
    for exp in explanations:
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

    print("SHAP Ranks and Pct Occurances one unrelated features:")
    e = experiment_summary(formatted_explanations, features)
    print(e)
    save_results(path_shap, e, 'shap')

    print("Fidelity:", round(adv_shap.fidelity(xtest), 2))

    print('---------------------')


if __name__ == "__main__":
    path = f"../datasets/german_lmodified_{seed}/"
    path1 = f"../datasets/german_smodified_{seed}/"

    experiment_main(path, path1, gender_indc, columns, xmin, xmax,load=False,load_dataset=False)
