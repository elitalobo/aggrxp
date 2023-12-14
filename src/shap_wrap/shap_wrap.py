#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## shap_wrap.py (reuses parts of the code of SHAP)
##
##  Created on: Sep 25, 2019
##      Author: Nina Narodytska
##      E-mail: narodytska@vmware.com
##

#
#==============================================================================
import json
import numpy as np
import xgboost as xgb
import math
import shap
import resource
np.random.seed(1)

#
#==============================================================================

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

def shap_call(xgb, sample = None, feats='all', nb_features_in_exp = None,attack=None):
    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    f2imap = {}
    for i, f in enumerate(xgb.feature_names):
        f2imap[f.strip()] = i

    if (sample is not None):
        if (nb_features_in_exp is None):
            nb_features_in_exp = len(sample)

        try:
            feat_sample  = np.asarray(sample, dtype=np.float32)
        except:
            print("Cannot parse input sample:", sample)
            exit()
        print("\n\n Starting SHAP explainer... \n Considering a sample with features:", feat_sample)
        if not (len(feat_sample) == len(xgb.X_train[0])):
            print("Unmatched features are not supported: The number of features in a sample {} is not equal to the number of features in this benchmark {}".format(len(feat_sample), len(xgb.X_train[0])))
            exit()

        # compute boost predictions
        feat_sample_exp = np.expand_dims(feat_sample, axis=0)
        feat_sample_exp = xgb.transform(feat_sample_exp)

        y_pred = xgb.model.predict(feat_sample_exp)[0]
        y_pred_prob = xgb.model.predict_proba(feat_sample_exp)[0]




        def attack_predict_fn(x):
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape((1, -1))
            probs = xgb.model.predict_proba(xgb.transform(x)).astype(float)
            class_id = xgb.predict(x).astype(float).flatten()
            biased = [r for r in xgb.biasLayer]
            unbiased = [r for r in xgb.unbiasLayer]

            adv_targets = list(set({r['class'] for r in (biased + unbiased)}))
            n_classes = len(adv_targets)

            class_probs = np.zeros((x.shape[0],n_classes))
            class_id = class_id.astype(int)
            # class_probs[np.arange(len(class_id)), class_id] = 1.0
            # assert(np.all(np.sum(class_probs,axis=1)==1))

            return class_id
            # race_indc = xgb.feature_names.index('race')
            # unrelated_indcs = xgb.feature_names.index('unrelated_column_one')
            #
            # feat_sample_exp = xgb.transform(x)
            #
            # not_ood = xgb.model.predict(feat_sample_exp)
            #
            # biased = (x[:, race_indc] < 0).astype('int')
            # unbiased = (x[:, unrelated_indcs] < 0).astype('int')
            # one_hot_biased = one_hot_encode(biased)
            # one_hot_unbiased = one_hot_encode(unbiased)
            # sol = np.where(not_ood==1, biased,
            #                unbiased)


            # return sol.flatten()



        # return class_probs

        if attack:
            # do not transform, already getting transformed in predict class
            feat_sample_exp = np.expand_dims(feat_sample, axis=0)
            y_pred = xgb.predict(feat_sample_exp)[0]
            y_pred_prob = one_hot_encode(attack_predict_fn(feat_sample_exp))[0]
            background_distribution = shap.kmeans(xgb.X_train, 10)

            explainer = shap.KernelExplainer(attack_predict_fn, background_distribution)

        else:
            explainer = shap.TreeExplainer(xgb.model)
        shap_values = explainer.shap_values(feat_sample_exp)

        # No need to pass dataset as it is recored in model
        # https://shap.readthedocs.io/en/latest/

        shap_values_sample = shap_values[-1]

        transformed_sample = feat_sample_exp[-1]


        # we need to sum values per feature
        # https://github.com/slundberg/shap/issues/397
        sum_values = []
        if (xgb.use_categorical):
            p = 0
            for f in xgb.categorical_features:
                nb_values = len(xgb.categorical_names[f])
                sum_v = 0
                for i in range(nb_values):
                    sum_v = sum_v + shap_values_sample[p+i]
                p = p + nb_values
                sum_values.append(sum_v)
        else:
            sum_values = shap_values_sample
        expl = []

        # choose which features in the explanation to focus on
        if feats in ('p', 'pos', '+'):
            feats = 1
        elif feats in ('n', 'neg', '-'):
            feats = -1
        else:
            feats = 0

        print("\t \t Explanations for the winner class", y_pred, " (xgboost confidence = ", y_pred_prob[int(y_pred)], ")")
        print("base_value = {}, predicted_value = {}".format(explainer.expected_value, np.sum(sum_values) + explainer.expected_value))

        abs_sum_values = np.abs(sum_values)
        sorted_by_abs_sum_values =np.argsort(-abs_sum_values)

        all_expls=[]

        for k1, v1 in enumerate(sorted_by_abs_sum_values):

            k = v1
            v = sum_values[v1]

            all_expls.append((f2imap[xgb.feature_names[k]], xgb.feature_names[k], v))


            if (feats == 1 and v < 0) or (feats == -1 and v >= 0):
                continue

            expl.append(f2imap[xgb.feature_names[k]])
            print("id = {}, name = {}, score = {}".format(f2imap[xgb.feature_names[k]], xgb.feature_names[k], v))

            if (len(expl) ==  nb_features_in_exp):
                break

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        print('  time: {0:.2f}'.format(timer))

        return all_expls, y_pred, y_pred_prob