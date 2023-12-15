#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xreason.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev, Nina Narodytska
##      E-mail: alexey.ignatiev@monash.edu, narodytska@vmware.com
##

#
import resource
#==============================================================================
# from __future__ import print_function
from data import Data
from anchor_wrap import anchor_call
from lime_wrap import lime_call
from shap_wrap import shap_call
from options import Options
import joblib
import numpy as np
import os
import sys
from xgbooster import XGBooster, preprocess_dataset


#
#==============================================================================
def show_info():
    """
        Print info message.
    """

    print('c XReason: reasoning about explanations')
    print('c author(s): Alexey Ignatiev    [email:alexey.ignatiev@monash.edu]')
    print('c            Joao Marques-Silva [email:joao.marques-silva@irit.fr]')
    print('c            Nina Narodytska    [email:narodytska@vmware.com]')
    print('')

def multi_run_wrapper(args):
   return compute(*args)


def compute(point,options,idx,true_y):


    point_ = [round(float(x),2) for x in point]
    if options.uselime or options.useanchor or options.useshap:
        xgb = XGBooster(options, from_model=options.files[0], categorical_features=categorical_feature_names)
    else:
        # abduction-based approach requires an encoding
        xgb = XGBooster(options, from_model=options.files[0], categorical_features=categorical_feature_names)


    if options.explain:
        options.explain = point_


    if options.encode:
        # if not xgb:

        # encode it and save the encoding to another file
        xgb.encode(test_on=point_)


    feat_sample_exp = np.expand_dims(point_, axis=0)
    feat_sample_tr = xgb.transform(feat_sample_exp)

    if options.attack:
        y_pred = xgb.predict(feat_sample_exp)[0]
    else:
        y_pred = xgb.model.predict(feat_sample_tr)[0]

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime


    expl_ = xgb.explain(point_,
                       use_lime=lime_call if options.uselime else None,
                       use_anchor=anchor_call if options.useanchor else None,
                       use_shap=shap_call if options.useshap else None,
                       nof_feats=options.limefeats,attack=options.attack)


    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
    if options.uselime==True or options.useshap==True:
        expl = expl_[0]
        y_pred =  expl_[1]

    else:
        expl = expl_

    if (options.uselime or options.useanchor or options.useshap) and options.validate:
        xgb.validate(options.explain, expl)

    return (point,idx,expl,y_pred,true_y,time)

#


#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)


    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

    if (options.preprocess_categorical):
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    if options.files:
        xgb = None

        if options.train:
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)



            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()
            res=[train_accuracy,test_accuracy,options.files[0]]
            f_name = options.files[0].split(".csv")[0] + "_fidelity.pkl"
            joblib.dump(res,f_name)


        categorical_feature_names = []

        # read a sample from options.explain

        if options.explain:

            options.explain = [float(v.strip()) for v in options.explain.split(',')]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0],categorical_features=categorical_feature_names)

            try:
                # encode it and save the encoding to another file
                xgb.encode(test_on=options.explain)
            except:

                print("could not encode sample")


        if options.attack:


            try:
                f = options.files[1].split("/")[-1].strip(".csv")[0]
            except:
                f = options.files[0].split("/")[-1].strip(".csv")[0]

            idx = 0
            all_expl=[]

            # if options.attack:
            if "compas" in options.files[0]:
                categorical_feature_names = ['unrelated_column_one', 'unrelated_column_two',
                 'c_charge_degree_F','c_charge_degree_M',
                 'two_year_recid', 'race', "sex_Male",
                "sex_Female"]

            elif "german" in options.files[0]:
                categorical_feature_names= categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
           'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
           'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
           'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
           'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled','LoanRateAsPercentOfIncome']



            else:
                print("no categorical features found in lime attack")

            if not xgb:
                if options.uselime or options.useanchor or options.useshap:
                    xgb = XGBooster(options, from_model=options.files[0],categorical_features=categorical_feature_names)
                else:
                    # abduction-based approach requires an encoding
                    xgb = XGBooster(options, from_encoding=options.files[0],categorical_features=categorical_feature_names)

            if len(options.files) >= 2:
                data = Data(filename=options.files[1], mapfile=options.mapfile,
                            separator=options.separator,
                            use_categorical=options.use_categorical)

                xgb_test = XGBooster(options, from_data=data, categorical_features=categorical_feature_names)

            else:
                data = Data(filename=options.files[0], mapfile=options.mapfile,
                            separator=options.separator,
                            use_categorical=options.use_categorical)
                xgb_test = xgb

            data_name = xgb.basename.split("/")[-1]
            type = "mwc"
            if options.uselime:
                type = "lime"
            if options.useshap:
                type = "shap"

            dirname = "data/" + data_name
            if os.path.exists("data/") == False:
                os.mkdir("data/")
            if os.path.exists(dirname) is False:
                os.mkdir(dirname)
            fname = dirname + "/" + "imp.pkl"

            options.limefeats = len(data.names) - 1

            points = []
            result = []

            for point in xgb_test.X:


                for jdx in range(int(xgb_test.weights[idx])):
                    points.append((point,options,idx,fname,dirname,xgb_test.Y[idx]))
                    result.append(multi_run_wrapper((point,options,idx,xgb_test.Y[idx])))

                idx+=1

                if idx%20==0:
                    joblib.dump(result, dirname + "/" + type + "_expls.pkl")
                    joblib.dump(points, dirname + "/" + type + "_points.pkl")

            all_expl = result
            joblib.dump(all_expl,dirname + "/"  + type+ "_expls.pkl")
            joblib.dump(points, dirname + "/" + type + "_points.pkl")

        else:
            try:
                f = options.files[1].split("/")[-1].strip(".csv")[0]
            except:
                f = options.files[0].split("/")[-1].strip(".csv")[0]

            idx = 0
            all_expl = []


            if not xgb:
                if options.uselime or options.useanchor or options.useshap:
                    xgb = XGBooster(options, from_model=options.files[0],
                                    categorical_features=None)
                else:
                    # abduction-based approach requires an encoding
                    xgb = XGBooster(options, from_encoding=options.files[0],
                                    categorical_features=None)

            if len(options.files) >= 2:
                data = Data(filename=options.files[1], mapfile=options.mapfile,
                            separator=options.separator,
                            use_categorical=options.use_categorical)

                xgb_test = XGBooster(options, from_data=data, categorical_features=None)

            else:
                data = Data(filename=options.files[0], mapfile=options.mapfile,
                            separator=options.separator,
                            use_categorical=options.use_categorical)
                xgb_test = xgb

            data_name = xgb.basename.split("/")[-1]
            type = "mwc"
            if options.uselime:
                type = "lime"
            if options.useshap:
                type = "shap"

            dirname = "data/" + data_name
            if os.path.exists("data/") == False:
                os.mkdir("data/")
            if os.path.exists(dirname) is False:
                os.mkdir(dirname)
            fname = dirname + "/" + "imp.pkl"

            options.limefeats = len(data.names) - 1

            points = []
            result = []

            for point in xgb_test.X:

                for jdx in range(int(xgb_test.weights[idx])):
                    points.append((point, options, idx, fname, dirname, xgb_test.Y[idx]))
                    result.append(multi_run_wrapper((point, options, idx, xgb_test.Y[idx])))

                idx += 1

                if idx % 20 == 0:
                    joblib.dump(result, dirname + "/" + type + "_expls.pkl")
                    joblib.dump(points, dirname + "/" + type + "_points.pkl")

            all_expl = result
            joblib.dump(all_expl, dirname + "/" + type + "_expls.pkl")
            joblib.dump(points, dirname + "/" + type + "_points.pkl")
