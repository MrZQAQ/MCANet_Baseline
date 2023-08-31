#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-07-13 13:32
LastEditTime: 2022-07-13 13:50
LastEditors: MrZQAQ
Description: Custom DataSets
FilePath: /TransformerCPI_mod/dataset.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import torch
import numpy as np
from torch.utils.data import Dataset
from word2vec import seq_to_kmers, get_protein_embedding
from mol_featurizer import mol_features
from gensim.models import Word2Vec

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


class collater():
    '''
    Description: DataLoader 的 collate_fn, 包装成类以加载 word2vec 模型
    '''
    def __init__(self):
        self.vec_model = Word2Vec.load("word2vec_30_mod.model")

    def __call__(self,batch_data):
        compounds, adjacencies,proteins,interactions = [], [], [], []
        for no, data in enumerate(batch_data):
            smiles, sequence, interaction = data.strip().split(" ")
            atom_feature, adj = mol_features(smiles)
            protein_embedding = get_protein_embedding(self.vec_model, seq_to_kmers(sequence))
            label = np.array(interaction,dtype=np.float32)

            atom_feature = torch.FloatTensor(atom_feature)
            adj = torch.FloatTensor(adj)
            protein = torch.FloatTensor(protein_embedding)
            label = torch.LongTensor(label)

            compounds.append(atom_feature)
            adjacencies.append(adj)
            proteins.append(protein)
            interactions.append(label)
        
        return list(zip(compounds, adjacencies, proteins, interactions))
        