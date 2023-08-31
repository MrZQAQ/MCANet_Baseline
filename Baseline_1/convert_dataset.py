#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-06-03 12:50
LastEditTime: 2022-06-03 16:47
LastEditors: MrZQAQ
Description: Dataset convert
FilePath: /DeepConv-DTI/convert_dataset.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import sys


def smilesToMorginFingerprintString(smiles:str) -> str:
    m1 = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,2)
    text_list = re.findall(".{1}",fp1.ToBitString())
    text = "\t".join(text_list)
    return text

def saveDrugtoDf(dataframe:pd.DataFrame, drugname:str, drugsmiles:str, drugmorgin:str) -> pd.DataFrame:
    newdf = pd.DataFrame([[drugname,drugsmiles,drugmorgin]],columns=('Compound_ID','SMILES','morgan_fp_r2'))
    dataframe = pd.concat([dataframe,newdf],ignore_index=True)
    return dataframe

def saveProteintoDf(dataframe:pd.DataFrame, proteinname:str, proteinaa:str) -> pd.DataFrame:
    newdf = pd.DataFrame([[proteinname,proteinaa]],columns=['Protein_ID','Sequence'])
    dataframe = pd.concat([dataframe,newdf],ignore_index=True)
    return dataframe

def savedtitoDf(dataframe:pd.DataFrame, drugname:str, proteinname:str, label:str) -> pd.DataFrame:
    newdf = pd.DataFrame([[drugname,proteinname,label]],columns=('Compound_ID','Protein_ID','Label'))
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
    drugdf = pd.DataFrame(columns=('Compound_ID','SMILES','morgan_fp_r2'))
    proteindf = pd.DataFrame(columns=('Protein_ID','Sequence'))
    dtidf = pd.DataFrame(columns=('Compound_ID','Protein_ID','Label'))
    for line in lines:
        linelist = line.strip().split()
        drugname = linelist[0]
        drugsmiles = linelist[2]
        proteinname = linelist[1]
        proteinaa = linelist[3]
        label = linelist[4]
        if drugname not in drug_visted:
            drug_visted.append(drugname)
            drugdf = saveDrugtoDf(drugdf,drugname,drugsmiles,smilesToMorginFingerprintString(drugsmiles))
        if proteinname not in protein_visited:
            protein_visited.append(proteinname)
            proteindf = saveProteintoDf(proteindf,proteinname,proteinaa)
        dtidf = savedtitoDf(dtidf,drugname,proteinname,label)
    return drugdf,proteindf,dtidf

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
    drugdf,proteindf,dtidf = processlist(listtest)
    drugdf.to_csv(f'data/{dataset}/test/drug.csv')
    proteindf.to_csv(f'data/{dataset}/test/protein.csv')
    dtidf.to_csv(f'data/{dataset}/test/dti.csv')
    print('done.')
    k_fold = 5
    for i in range(k_fold):
        print(f'*****     start fold  {i+1}     *****')
        print('split train and val...')
        listtrain,listval = get_kfold_data(i,lines[0:test_split_pos],k_fold)
        print('process train dataset list...')
        drugdf,proteindf,dtidf = processlist(listtrain)
        drugdf.to_csv(f'data/{dataset}/train/{i+1}/drug.csv')
        proteindf.to_csv(f'data/{dataset}/train/{i+1}/protein.csv')
        dtidf.to_csv(f'data/{dataset}/train/{i+1}/dti.csv')
        print('done.')
        print('process validation dataset list...')
        drugdf,proteindf,dtidf = processlist(listval)
        drugdf.to_csv(f'data/{dataset}/validation/{i+1}/drug.csv')
        proteindf.to_csv(f'data/{dataset}/validation/{i+1}/protein.csv')
        dtidf.to_csv(f'data/{dataset}/validation/{i+1}/dti.csv')
        print('done.')
        
dataset  = sys.argv[1]
assert dataset in ['Davis','KIBA','DrugBank']
main(dataset)