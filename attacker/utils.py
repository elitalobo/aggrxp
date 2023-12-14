import json
import numpy as np
import pandas as pd
import random
from argparser import parse
args = parse()
random.seed(args.seed)
np.random.seed(args.seed)
seed= args.seed
class Params():
    """Parameters object taken from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    
    Parameters
    ----------
    json_path : string

    Returns
    ----------
    Parameters object
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def one_hot_encode(y):
    """ One hot encode y for binary features.  We use this to get from 1 dim ys to predict proba's.
    This is taken from this s.o. post: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

    Parameters
    ----------
    y : np.ndarray

    Returns
    ----------
    A np.ndarray of the one hot encoded data.
    """
    y_hat_one_hot = np.zeros((len(y), 2))
    y_hat_one_hot[np.arange(len(y)), y] = 1
    return y_hat_one_hot

def rank_features(explanation):
    """ Given an explanation of type (name, value) provide the ranked list of feature names according to importance

    Parameters
    ----------
    explanation : list

    Returns
    ----------
    List contained ranked feature names
    """

    ordered_tuples = sorted(explanation, key=lambda x : abs(x[1]), reverse=True)
    results = [tup[0] if tup[1] != 0 else ("Nothing shown",0) for tup in ordered_tuples]
    return results
def rank_features1(explanation):
    """ Given an explanation of type (name, value) provide the ranked list of feature names according to importance

    Parameters
    ----------
    explanation : list

    Returns
    ----------
    List contained ranked feature names
    """
    # from scipy.stats import rankdata
    # ranks = rankdata(explanation, method='min')
    # return ranks
    ordered_tuples = sorted(explanation, key=lambda x : abs(x[1]), reverse=True)
    ranks = []
    r=0
    score = ordered_tuples[0][1]
    for tuple in ordered_tuples:
        if tuple[1]!=score:
            score = tuple[1]
            r+=1

        ranks.append((r,tuple[0],tuple[1]))


    # results = [tup[0] if tup[1] != 0 else ("Nothing shown",0) for tup in ordered_tuples]

    return ranks


def get_rank_map(ranks, to_consider):
    """ Give a list of feature names in their ranked positions, return a map from position ranks
    to pct occurances.

    Parameters
    ----------
    ranks : list
    to_consider : int

    Returns
    ----------
    A dictionary containing the ranks mapped to the uniques.
    """
    unique = {i+1 : [] for i in range(len(ranks))}

    for i, rank in enumerate(ranks):
        for unique_rank in np.unique(rank):
            unique[i+1].append((unique_rank, np.sum(np.array(rank) == unique_rank) / to_consider))

    return unique

def experiment_summary(explanations, features):
    """ Provide a high level display of the experiment results for the top three features.
    This should be read as the rank (e.g. 1 means most important) and the pct occurances
    of the features of interest.

    Parameters
    ----------
    explanations : list
    explain_features : list
    bias_feature : string

    Returns 
    ----------
    A summary of the experiment
    """
    # features_of_interest = explain_features + [bias_feature]   
    top_features = [[], [], []]

    # sort ranks into top 3 features
    for exp in explanations:
        ranks = rank_features(exp)
        ranks1 = rank_features1(exp)

        for tuple in ranks1:
            if tuple[0]<3:
                r = tuple[0]
                f = tuple[1]
                if "=" in f:
                    feat = f.split("=")[0]
                else:
                    feat = f
                top_features[r].append(feat)

        # for i in range(3):
        #     for f in features + ["Nothing shown"]:
        #         if f in ranks[i]:
        #             top_features[i].append(f)

    return get_rank_map(top_features, len(explanations))


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def rank_top_features(X,y,cols,categorical,not_categorical):

    top_cat = []
    top_noncat = []


    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0,n_estimators=80)
    forest.fit(X, y)

    importances = forest.feature_importances_

    ranks = np.argsort(importances)
    print(np.sort(-1.0*importances))


    for f in ranks:
        feature = cols[f]
        if feature in categorical:
            top_cat.append(f)
        else:
            top_noncat.append(f)
    print("non cat",top_noncat[:7])
    print("top cat",top_cat[:4])
    selected = list(top_noncat[:7]) + list(top_cat[:4])
    X_new = X[:,selected]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_new, y)
    print(forest.score(X_new,y))

    return selected




