import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import math

def get_string_form(expl_indices):
    e = sorted(expl_indices)
    e_str = [str(x) for x in e]
    e_str = "-".join(e_str)
    return e_str


class ExplanationAggregator():
    def __init__(self,expls,n_features, preamble,path, map_f):
        self.expls = expls
        self.preamble = preamble
        self.n_features = n_features
        self.basename = path.split("/")[1]

        self.map_f = map_f



    def get_indice_values(self):
        self.holler={}
        self.deegan={}
        self.resp={}
        self.res = []
        self.res_holler = []
        self.labels={}
        self.resp_rank = {}
        self.deegan_rank = {}
        self.holler_rank = {}
        self.points={}


        for expl in self.expls:
            explanations = {}


            point, idx,explanation, ypred, true_y, time = expl
            es=[]
            for e in explanation:
                es_str = get_string_form(e)

                explanations[es_str] = True

                es.append(e)


            r, h, d, p = self.compare_index(es)
            self.resp[idx]=r
            self.points[idx]=point

            self.deegan[idx] = d
            self.deegan_rank = np.argsort(d)
            self.holler_rank = np.argsort(h)
            self.resp_rank =  np.argsort(r)
            self.holler[idx] = h
            self.labels[idx] = p

            abs_vals = np.max(np.abs(r - h) + np.abs(r - d) + np.abs(
                h- d))
            values = np.vstack((r,h,d))
            self.res.append(tuple([values, abs_vals, idx]))

            abs_vals = np.max(np.abs(
                h- d))
            self.res_holler.append(tuple([values, abs_vals, idx]))

    def responsibility_index(self, expls, sort=False):
        n_features = self.n_features
        importances = {}

        for expl in expls:
            for f in expl:
                if importances.get(f) is None:
                    importances[f] = len(expl)
                else:
                    importances[f] = min(importances[f], len(expl))
        for idx in range(n_features):
            if importances.get(idx) is None:
                importances[idx] = 0.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            if val == 0:
                ranks.append(tuple([key, 0.0]))
            else:

                ranks.append(tuple([key, 1.0 / val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)

        return ranks


    def deegan_packel(self, expls, sort=False):
        n_features = self.n_features
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:

                if len(expl) != 0:
                    importances[f] += 1.0 / len(expl)
        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks


    def holler_packel(self, expls, sort=False):
        n_features = self.n_features
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:
                importances[f] += 1.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks


    def normalize(self, ranks):
        scores = []
        for rank in ranks:
            scores.append(rank[1])

        scores = np.array(scores)

        if np.sum(scores) == 0:
            normalized_scores = scores
        else:
            normalized_scores = np.round(scores / np.sum(scores), 2)

        return normalized_scores


    def compare_index(self, expls, writer=None):
        response_rank = self.responsibility_index(expls,sort=False)
        holler_rank = self.holler_packel(expls, sort=False)
        deegan_rank = self.deegan_packel(expls, sort=False)

        norm_response = self.normalize(response_rank)
        norm_holler = self.normalize(holler_rank)
        norm_deegan = self.normalize(deegan_rank)


        return norm_response, norm_holler, norm_deegan, self.preamble




def find_top(scores, p):
    scores = np.array(scores)
    if p==0:

        ranks = np.argsort(scores)
    else:
        ranks = np.argsort(-1.0 * scores)
    return ranks

def find_imp(scores,p):
    scores = np.array(scores)
    if p==0:
        flags = scores<0
    else:
        flags = scores>=0
    lent = len(flags)
    imp=[]
    for idx in range(lent):
        if flags[idx]==1:
            imp.append(idx)


    return imp





def check_superset(features,expl_ids):
    for expl in expl_ids:
        bool = True
        for id in expl:
            if id not in features:
                bool = False
        if bool == True:
            return True
    return False




def get_complementary(ids, N):
    c=[]
    for idx in range(N):
        if idx not in ids:
            c.append(idx)
    return c






def compute_ranks(top_indices, labels):
    lime_features={}
    d_features={}
    for key, value in top_indices.items():
        all_label = labels[key]
        indices = value['deegan-packel']
        l_indices = value['lime']
        rank=0
        for idx in indices:
            label = all_label[idx]
            if d_features.get(label) is None:
                d_features[label]=np.zeros(100)
            d_features[label][rank]+=1

            rank+=1

        rank=0
        for idx in l_indices:
            label = all_label[idx]
            if lime_features.get(label) is None:
                lime_features[label] = np.zeros(100)
            lime_features[label][rank] += 1

            rank += 1




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
    top_features = [[], [], [], [], []]


    for exp in explanations:
        ranks = rank_features(exp)
        for tuple in ranks:
            if tuple[0]<5:
                r = tuple[0]
                top_features[r].append(tuple[1])

    return get_rank_map(top_features, len(explanations))


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
    ranks = []
    r=0
    score = ordered_tuples[0][1]
    for tuple in ordered_tuples:
        if tuple[1]!=score:
            score = tuple[1]
            r+=1

        ranks.append((r,tuple[0],tuple[1]))

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


def get_indices_maps(features):
    map_f = {}
    map_b = {}
    for idx in range(len(features)):
        map_f[idx] = features[idx]
        map_b[features[idx]] = idx

    return map_f, map_b




def save_results(path,results,filename=None):
	joblib.dump(results,path + filename + ".pkl")


def compute_lime_explanations(dataset_name,expl_path,data_path="datasets/"):
    expls = joblib.load(expl_path)
    n = len(expls)
    formatted_explanations = []
    dataset_path = data_path + dataset_name + "/" + dataset_name + "_test.csv"
    df = pd.read_csv(dataset_path)

    features = df.columns[:-1]
    map_feature={}
    map_index = {}
    idx=0
    for f in features:
        map_index[idx]=f
        map_feature[f]=idx
        idx+=1

    for idx in range(n):
        expl = expls[idx][2]
        final_feature_scores=[]
        feature_scores=np.zeros(len(features))
        for item in expl:
            feature = item[0].split("=")[0]
            value = item[1]
            index = map_feature[feature]
            feature_scores[index]+=value


        for jdx in range(len(features)):
            final_feature_scores.append((features[jdx],feature_scores[jdx]))

        formatted_explanations.append(final_feature_scores)
    e = experiment_summary(formatted_explanations, features)
    return e





def compute_shap_explanations(dataset_name,expl_path,data_path="datasets/"):
    expls = joblib.load(expl_path)
    n = len(expls)
    formatted_explanations = []

    dataset_path = data_path + dataset_name + "/" + dataset_name + "_test.csv"
    df = pd.read_csv(dataset_path)
    features = df.columns[:-1]


    for idx in range(n):
        expl = expls[idx][2]
        final_feature_scores = []
        for item in expl:
            final_feature_scores.append((item[1],item[2]))



        formatted_explanations.append(final_feature_scores)
    e = experiment_summary(formatted_explanations, features)
    return e



def compute_abductive_explanations(dataset_name,expl_path,data_path="datasets/"):


    dataset_path = data_path + dataset_name + "/" + dataset_name + "_test.csv"
    df = pd.read_csv(dataset_path)
    features = df.columns[:-1]
    map_f, map_b = get_indices_maps(features)
    expls = joblib.load(expl_path)

    explAgg = ExplanationAggregator(expls,len(features),features,expl_path,map_b)
    explAgg.get_indice_values()

    formatted_explanations = []
    for idx, exp in explAgg.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_resp= experiment_summary(formatted_explanations, features)

    formatted_explanations = []
    for idx, exp in explAgg.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_holler = experiment_summary(formatted_explanations, features)


    formatted_explanations = []
    for idx, exp in explAgg.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_deegan = experiment_summary(formatted_explanations, features)
    return e_resp, e_holler, e_deegan



def compute_time_abductive_explanations(dataset_name,expl_path,data_path="datasets/"):


    dataset_path = data_path + dataset_name + "/" + dataset_name + "_test.csv"
    df = pd.read_csv(dataset_path)
    features = df.columns[:-1]
    map_f, map_b = get_indices_maps(features)
    expls = joblib.load(expl_path)

    explAgg = ExplanationAggregator(expls,len(features),features,expl_path,map_b)
    explAgg.get_indice_values()

    formatted_explanations = []
    for idx, exp in explAgg.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_resp= experiment_summary(formatted_explanations, features)

    formatted_explanations = []
    for idx, exp in explAgg.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_holler = experiment_summary(formatted_explanations, features)


    formatted_explanations = []
    for idx, exp in explAgg.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    e_deegan = experiment_summary(formatted_explanations, features)
    return e_resp, e_holler, e_deegan



def compute_abductive_expls(dataset_name,expl_path,data_path="datasets/"):


    dataset_path = data_path + dataset_name + "/" + dataset_name + "_test.csv"
    df = pd.read_csv(dataset_path)
    features = df.columns[:-1]
    map_f, map_b = get_indices_maps(features)
    expls = joblib.load(expl_path)
    print(len(expls))

    explAgg = ExplanationAggregator(expls,len(features),features,expl_path,map_b)
    explAgg.get_indice_values()

    num_points = len(list(explAgg.points.keys()))
    num_features = len(explAgg.points[0])
    for idx in range(num_points):
        print("Explaining point:",explAgg.points[idx])

        resp_expl = [(features[i], explAgg.resp[idx][i]) for i in range(num_features)]
        hollder_expl = [(features[i], explAgg.holler[idx][i]) for i in range(num_features)]
        deegan_expl = [(features[i], explAgg.deegan[idx][i]) for i in range(num_features)]

        print('Feature importance weights based on Responsibility index: ',resp_expl)
        print('Feature importance weights based on Holler-Packel indec=x: ', hollder_expl)
        print('Feature importance weights based on Deegan-Packel index: ', deegan_expl)



