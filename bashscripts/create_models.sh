#!/bin/bash

python  src/xreason.py -t -n 100 datasets/compas_ood/compas_ood.csv
python src/xreason.py -t -n 100 datasets/compas_ood1/compas_ood1.csv
python src/xreason.py -t -n 100 datasets/compas_shapood/compas_shapood.csv
python src/xreason.py -t -n 100 datasets/compas_shapood1/compas_shapood1.csv
python src/xreason.py -t -n 50 datasets/german_lmodified/german_lmodified.csv
python src/xreason.py -t -n 50 datasets/german_smodified/german_smodified.csv

