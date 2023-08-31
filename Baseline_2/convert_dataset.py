#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-06-03 12:50
LastEditTime: 2022-07-10 09:43
LastEditors: MrZQAQ
Description: Dataset convert
FilePath: /TransformerCPI_mod/convert_dataset.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import sys
from tqdm import tqdm

def main(dataset = 'Davis'):
    filepath = f'./data/{dataset}.txt'
    print('read file...')
    with open(filepath,'r') as f:
        lines = f.readlines()
    newlines = []
    print('processing...')
    for line in tqdm(lines):
        linelist = line.replace('\n', '').replace('\r', '').strip().split()
        del linelist[0:2]
        newline = str(linelist[0]) + ' ' + str(linelist[1]) + ' ' + str(linelist[2]) + '\n'
        newlines.append(newline)
    with open(f'./data/{dataset}_mod.txt','w') as f:
        print('writting...')
        for item in tqdm(newlines):
        # for item in newlines:
            f.writelines(str(item))
            # print(item)
        
dataset  = sys.argv[1]
assert dataset in ['Davis','KIBA','DrugBank']
main(dataset)

