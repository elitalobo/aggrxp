import os

import sys

import pandas as pd

from utils import *

if __name__=='__main__':
    expl_path = sys.argv[1]
    dataset_name = sys.argv[2]

    if "mwc_expls" in expl_path:


        compute_abductive_expls(dataset_name,expl_path,data_path="datasets/")


    else:

        print("Not a valid explanations file")