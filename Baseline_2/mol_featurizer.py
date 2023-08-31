# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 14:19
@author: LiFan Chen
@Filename: mol_featurizer.py
@Software: PyCharm
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

num_atom_feat = 34

def split_data(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency,dtype=np.float32)


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    del mol
    return atom_feat, adj_matrix

def shuffle_dataset(dataset, seed):
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)
    # np.random.seed(seed)
    # np.random.shuffle(dataset)
    return dataset

if __name__ == "__main__":

    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    import os
    import torch
    import pickle
    import sys
    from tqdm import tqdm

    SEED=1234

    DATASET = sys.argv[1]

    assert DATASET in ["DrugBank", "KIBA", "Davis"]
    print("Process " + DATASET)
    if DATASET == "DrugBank":
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "Davis":
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "KIBA":
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")

    split_pos = len(data_list) - int(len(data_list) * 0.2)
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)
    train_and_vaild_dataset = data_list[0:split_pos]
    test_dataset = data_list[split_pos:-1]
    del data_list,split_pos
    
    model = Word2Vec.load("word2vec_30.model")
    total = len(train_and_vaild_dataset)
    part = 5
    splitpos = total // part
    temp = []
    for i in range(part):
        if i<4:
            temp.append(train_and_vaild_dataset[splitpos*i : splitpos*(i+1)])
        else:
            temp.append(train_and_vaild_dataset[splitpos*i : ])
    del train_and_vaild_dataset
    dataset = []
    for i,sublist in enumerate(temp):
        
        print(f'Part {i+1} of total {part}')
        compounds, adjacencies,proteins,interactions = [], [], [], []
        for no, data in tqdm(enumerate(sublist),total=len(sublist)):
            smiles, sequence, interaction = data.strip().split(" ")

            atom_feature, adj = mol_features(smiles)
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            label = np.array(interaction,dtype=np.float32)

            atom_feature = torch.FloatTensor(atom_feature)
            adj = torch.FloatTensor(adj)
            protein = torch.FloatTensor(protein_embedding)
            label = torch.LongTensor(label)

            compounds.append(atom_feature)
            adjacencies.append(adj)
            proteins.append(protein)
            interactions.append(label)
        dataset.extend(list(zip(compounds, adjacencies, proteins, interactions)))
        del compounds, adjacencies, proteins, interactions
    with open(f"./dataset/{DATASET}_train", "wb+") as f:
        for line in dataset:
            f.writelines()
        # print('The preprocess of ' + DATASET + ' dataset has finished!')


    N = len(test_dataset)
    compounds, adjacencies,proteins,interactions = [], [], [], []
    for no, data in tqdm(enumerate(test_dataset),total=N):
        smiles, sequence, interaction = data.strip().split(" ")

        atom_feature, adj = mol_features(smiles)
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
        label = np.array(interaction,dtype=np.float32)

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        protein = torch.FloatTensor(protein_embedding)
        label = torch.LongTensor(label)

        compounds.append(atom_feature)
        adjacencies.append(adj)
        proteins.append(protein)
        interactions.append(label)
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    with open(f"./dataset/{DATASET}_test.txt", "wb") as f:
        pickle.dump(dataset, fprotocol=5)
    # print('The preprocess of ' + DATASET + ' dataset has finished!')