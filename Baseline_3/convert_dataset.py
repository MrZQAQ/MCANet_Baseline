#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-06-03 12:50
LastEditTime: 2022-06-04 10:47
LastEditors: MrZQAQ
Description: Dataset convert
FilePath: /MolTrans/convert_dataset.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import sys

def savedtitoDf(dataframe:pd.DataFrame, drugname:str, proteinname:str, label:str,drugsmiles:str, proteinaa:str) -> pd.DataFrame:
    newdf = pd.DataFrame([[drugsmiles,proteinaa,label,drugname,proteinname]],columns=('SMILES','Target Sequence','Label','Drug Name','Target Name'))
    dataframe = pd.concat([dataframe,newdf],ignore_index=True)
    return dataframe

def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def processlist(lines:list):
    drug_visted = []
    protein_visited = []
    dtidf = pd.DataFrame(columns=('SMILES','Target Sequence','Label','Drug Name','Target Name'))
    for line in lines:
        linelist = line.strip().split()
        drugname = linelist[0]
        drugsmiles = linelist[2]
        proteinname = linelist[1]
        proteinaa = linelist[3]
        label = linelist[4]
        dtidf = savedtitoDf(dtidf,drugname,proteinname,label,drugsmiles,proteinaa)
    return dtidf

def shuffle_dataset(dataset, seed):
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)
    # np.random.seed(seed)
    # np.random.shuffle(dataset)
    return dataset

def main(dataset = 'Davis'):
    filepath = f'DataSets/{dataset}.txt'
    print('read file...')
    with open(filepath,'r') as f:
        lines = f.readlines()
    lines = shuffle_dataset(lines,114514)
    print('read done.')
    print('split test dataset...')
    totallen = len(lines)
    test_split_pos = totallen - int(totallen*0.2)
    listtest = lines[test_split_pos:]
    print('process test dataset list...')
    dtidf = processlist(listtest)
    dtidf.to_csv(f'dataset_mod/{dataset}/test.csv')
    print('done.')
    k_fold = 5
    for i in range(k_fold):
        print(f'*****     start fold  {i+1}     *****')
        print('split train and val...')
        listtrain,listval = get_kfold_data(i,lines[0:test_split_pos],k_fold)
        print('process train dataset list...')
        dtidf = processlist(listtrain)
        dtidf.to_csv(f'dataset_mod/{dataset}/train_{i+1}.csv')
        print('done.')
        print('process validation dataset list...')
        dtidf = processlist(listval)
        dtidf.to_csv(f'dataset_mod/{dataset}/val_{i+1}.csv')
        print('done.')
        
dataset  = sys.argv[1]
assert dataset in ['Davis','KIBA','DrugBank']
main(dataset)