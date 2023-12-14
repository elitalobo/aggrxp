#! /bin/bash


envs=("compas_ood" "compas_ood1" "compas_shapood" "compas_shapood1" "german_lmodified" "german_smodified" "compas_shapood" "compas_shapood1")
num=("100" "100" "100" "100" "50" "50" "50" "50")
for index in 0 1 2 3 4 5 6 7 8 9
do
    for id in 0 1 2 3 4 5 6 7
    do
      python src/xreason_test.py -l -L 100 --xnum 100 -a "datasets/${1}/config_num.yml" "temp/${1}_${3}/${1}_${3}_nbestim_${2}_maxdepth_3_testsplit_0.2.mod.pkl" "datasets/${1}_${3}/${1}_${3}_test.csv"
      python src/xreason_test.py -w -L 100 --xnum 100 -a "datasets/${1}/config_num.yml" "temp/${1}_${3}/${1}_${3}_nbestim_${2}_maxdepth_3_testsplit_0.2.mod.pkl" "datasets/${1}_${3}/${1}_${3}_test.csv"
      python src/xreason_test.py -v -e smt -s z3 --xnum 200 -a "datasets/${1}/config_num.yml" "temp/${1}_${3}/${1}_${3}_nbestim_${2}_maxdepth_3_testsplit_0.2.mod.pkl" "datasets/${1}_${3}/${1}_${3}_test.csv"

                         done
       done
