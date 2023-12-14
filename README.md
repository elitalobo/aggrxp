
# Axiomatic Aggregations of Abductive Explanations (Aggrxp)

This repository contains the code and experimental results accompanying the AAAI'24 paper - *'Axiomatic Aggregation of Abductive Explanations'*. This work extends `XReason` to support methods of aggregating abductive explanations for compute feature importance weights proposed in the AAAI'24 paper.

## Getting Started


To install the packages required to execute the code, run

```commandline
pip install -r requirements.txt
```

Download datasets and results from [data folder](https://drive.google.com/drive/folders/1EZBDXD58jnDHHHOd1I_sBZi4VQrgeKqr?usp=drive_link) and add them to the root folder.

Note that all the datasets are available in *datasets/* folder. Folders with names *\*_ood* contain datasets for Lime attack and folders with names *\*_shapood* contain datasets for SHAP attack.


## Usage

Aggrxp has a number of parameters, which can be set from the command line. To see the list of options, run (the executable script is located in [src](./src)):

```
$ xreason_agg.py -h
```

### Preparing a dataset

All datasets should be added to datasets/ folder. See datasets/compas_ood/ for an example dataset.

### Training an XGBoost Model

Before generating abductive explanations, an XGBoost model must be trained on the train dataset. 

To train an XGBoost model with T n_estimators, run

```commandline
python src/xreason.py -t -n T  datasets/traindataset/traindataset.csv
```

This will generate a *temp/someotherpath/\*_mod.pkl* file.

As an example, you can run the following command to train an XGBoost model with T=100 on *compas_ood* dataset.

```commandline
python src/xreason.py -t -n 100 datasets/compas_ood/compas_ood.csv
```

### Generating all abductive explanations

To compute M abductive explanations (AXps) for the predictions of the XGBoost model *temp/someotherpath/\*_mod.pkl* on a test dataset *datasets/dataset_name/testdataset.csv*, execute the following command.

```commandline
python src/xreason.py -v -e smt -s z3 --xnum M temp/someotherpath/*_mod.pkl datasets/dataset_name/testdataset.csv
```

For example, you can generate abductive explanations for the *compas_ood_small_test.csv* dataset by running the following command.

```commandline
python src/xreason.py -v -e smt -s z3 --xnum 100 -a datasets/compas_ood/config_num.yml temp/compas_ood/compas_ood_nbestim_100_maxdepth_3_testsplit_0.2.mod.pkl datasets/compas_ood/compas_ood_small_test.csv
```

We set the value of M sufficiently large to ensure that we get all the possible abductive explanations.
The above step will generate a *data/somepath1/mwc_expls.pkl* file containing all the abductive explanations. 


### Aggregating abductive explanations

To generate feature importance weights using our methods - Responsibility index, Holler-Packel Index, Deegan-Packel Index, run

```commandline
python scripts/compute_explanations_single.py data/somepath1/mwc_expls.pkl "dataset_name"
```

For example, for the compas_ood dataset, the path to explanations file is "data/compas_ood_nbestim_100_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2/mwc_expls.pkl" 
and the command for generating feature importance weights using our methods is

```commandline
python scripts/compute_explanations_single.py data/compas_ood_nbestim_100_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2/mwc_expls.pkl "compas_ood"
```

## Reproducing experimental results of the AAAI'24 paper


All the explanation files generated in our experiments can be found in *data/* folder.
All the trained models in our experiments can be found in *temp/* folder.

For our experiments, we generate datasets for training the OOD classifier for the attack experiments.
All the datasets for the LIME attack experiments (compas_ood, compas_ood1, shap_ood, shap_ood1, german_lmodified, german_smodified) and the SHAP attack experiments are stored in datasets/ folder.

To reproduce our results reported in the paper, execute the following steps:

1. Go to the `src` directory:

```commandline
cd src/
```

2. Train all the models at once:

```commandline
source bashscripts/create_models.sh
```

3. Given the trained models, run the experimentation script:

```
source run_experiments.sh
```

4. To compute the feature importance weights of biased and corresponding to LIME, SHAP, Responsibility Index, Deegan-Packel Index, and Holler-Packel Index, run

```commandline
python compute_attack_results.py
```

The script create_models.sh will train OOD classifiers (XGBoost models) for the compas and german credit datasets.
The script run_experiments.sh will invoke the adapted `XReason` library to generate all AXps, lIME and SHAP explanations on the test datasets and store them in data/ folder.
The script compute_attack_results.py will aggregate all the explanations in data/ and compute the feature importance weights of the biased and un-biased features.

## Citations

If `Aggrxp` has been significant to a project that leads to an academic publication, please, acknowledge that fact by citing it:

```
@misc{biradar2023model,
      title={Model Explanations via the Axiomatic Causal Lens}, 
      author={Gagan Biradar and Vignesh Viswanathan and Yair Zick},
      year={2023},
      eprint={2109.03890},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
