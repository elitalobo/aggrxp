#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xreason.py
##
##  Created on: 
##      Author: 
##      E-mail: 
##


#
#==============================================================================


from __future__ import print_function
from aggrxp.src.data import Data
from anchor_wrap import anchor_call
from lime_wrap import lime_call
from shap_wrap import shap_call
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset
import numpy  as np
import random
random.seed(1)
np.random.seed(1)

#
#==============================================================================
def show_info():
    """
        Print info message.
    """

    print('c XReason: reasoning about explanations')
    print('')


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

    log_dir = "logs/"
    if os.path.exists(log_dir)== False:
        os.mkdir(log_dir)

    if options.files:
        xgb = None

        if options.train:
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)

            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()

        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0])

            try:
                # encode it and save the encoding to another file
                xgb.encode(test_on=options.explain)
            except:
                print('Could not encode sample')


        if options.explain:
            if not xgb:
                if options.uselime or options.useanchor or options.useshap:
                    xgb = XGBooster(options, from_model=options.files[0])
                else:
                    # abduction-based approach requires an encoding
                    xgb = XGBooster(options, from_encoding=options.files[0])

            # checking LIME or SHAP should use all features
            if not options.limefeats:
                options.limefeats = len(data.names) - 1

            # explain using anchor or the abduction-based approach

            expl = xgb.explain(options.explain,
                    use_lime=lime_call if options.uselime else None,
                    use_anchor=anchor_call if options.useanchor else None,
                    use_shap=shap_call if options.useshap else None,
                    nof_feats = options.limefeats)


            if (options.uselime or options.useanchor or options.useshap) and options.validate:
                xgb.validate(options.explain, expl)


