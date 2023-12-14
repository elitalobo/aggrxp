#!/bin/bash



for i in 0 1 2 3 4 5 6 7 8 9
  do
    python german_modified_experiment.py --seed $i
  done

for i in 0 1 2 3 4 5 6 7 8 9
  do
    python compas_experiment.py --seed $i
  done