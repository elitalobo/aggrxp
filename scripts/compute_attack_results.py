import os

import pandas as pd

from utils import *


RESPONSIBILITY = "responsibility index"
HOLLER = "holler-packel index"
DEEGAN = "deegan-packel index"
LIME = "lime"
SHAP = "shap"

if __name__=='__main__':
    expl_dir = "data/"
    results={}
    for dir in os.listdir(expl_dir):
        words=dir.split("_")
        if words[2]=="nbestim":
            continue
        dataset_name = "_".join(words[:2])
        n_estimators = dir.split('nbestim')[1].split("_")[1]
        name = dataset_name + " n_estimators=" + n_estimators

        if results.get(name) is None:
            results[name]={}
        for f in os.listdir(expl_dir + dir):
            if "mwc_expls" in f:
                f_path = expl_dir + dir + "/" + f
                if results[name].get(RESPONSIBILITY) is None:
                    results[name][RESPONSIBILITY] = []
                    results[name][HOLLER] = []
                    results[name][DEEGAN] = []


                e_resp, e_holler, e_deegan = compute_abductive_explanations(dataset_name,f_path)
                results[name][RESPONSIBILITY].append(e_resp)
                results[name][HOLLER].append(e_holler)
                results[name][DEEGAN].append(e_deegan)

            elif "lime_expls" in f:
                f_path = expl_dir + dir + "/" + f
                lime_exp = compute_lime_explanations(dataset_name, f_path)
                if results[name].get(LIME) is None:
                    results[name][LIME]=[]
                results[name][LIME].append(lime_exp)



            elif "shap_expls" in f:
                f_path = expl_dir + dir + "/" + f
                shap_exp = compute_shap_explanations(dataset_name, f_path)
                if results[name].get(SHAP) is None:
                    results[name][SHAP] = []
                results[name][SHAP].append(shap_exp)



            else:
                continue

compas_imp_f = ['race','unrelated_column_one','unrelated_column_two']
german_imp_f= ['Gender', 'LoanRateAsPercentOfIncome']
methods_lime=[LIME, RESPONSIBILITY, HOLLER, DEEGAN]
methods_shap=[SHAP, RESPONSIBILITY, HOLLER, DEEGAN]

features_map={'race': 'Race', 'unrelated_column_one': 'UC1', 'unrelated_column_two': 'UC2', 'Gender': 'Gender', 'LoanRateAsPercentOfIncome':'LR'}

for key, value in results.items():
    if "compas" in key:
        imp_features = compas_imp_f
    else:
        imp_features = german_imp_f

    if "shap" in key or "smodified" in key:
        methods = methods_shap
    else:
        methods = methods_lime
    print("\n\nDataset: " + key)
    try:

        for f in imp_features:

            line =  features_map[f] + " "
            for method in methods:
                ress = results[key][method]

                for idx in range(3):
                    values=[]
                    for res in ress:
                        try:
                            vals = res[idx+1]
                        except:
                            print("####method", method)
                            print("###res", res)
                            # line += " & " + str(0.0)
                            values.append(0)
                            continue
                        found=False
                        for v in vals:
                            if v[0]==f:
                                values.append(v[1])
                                found=True
                                break
                        if found==False:
                            values.append(0.0)
                    # print("values",len(values))
                    line += "& " + str(np.round(np.mean(values),2)) + " ± " + str(np.round(2*np.std(values),2))


            print(line + " \\\\ ")



    except:
        pass

    print("\n\n")

