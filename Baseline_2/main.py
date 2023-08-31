# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/25 10:03
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import time
from model import *
import timeit
import pickle
import sys

from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import os
import torch
import pickle
import sys
from tqdm import tqdm
from mol_featurizer import mol_features
from dataset import CustomDataSet, collater

def shuffle_dataset(dataset, seed):
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)
    # np.random.seed(seed)
    # np.random.shuffle(dataset)
    return dataset

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

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def show_result(DATASET,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The model's results:")
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

if __name__ == "__main__":
    SEED = 114514
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    DATASET = sys.argv[1]

    # with open(f'./dataset/{DATASET}_train',"rb") as f:
    #     data = pickle.load(f)
    # dataset = shuffle_dataset(data, 1234)
    # dataset_train, dataset_val = split_dataset(dataset, 0.8)
    # with open(f'./dataset/{DATASET}_train',"rb") as f:
    #     data = pickle.load(f)
    # dataset_test = shuffle_dataset(data, 1234)
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
    dataset_train_and_vaild = data_list[0:split_pos]
    dataset_test = data_list[split_pos:-1]

    # vec_model = Word2Vec.load("word2vec_30_mod.model")

    # compounds, adjacencies,proteins,interactions = [], [], [], []
    # for no, data in tqdm(enumerate(dataset_train_and_vaild),total=len(dataset_train_and_vaild)):
    #     smiles, sequence, interaction = data.strip().split(" ")

    #     atom_feature, adj = mol_features(smiles)
    #     protein_embedding = get_protein_embedding(vec_model, seq_to_kmers(sequence))
    #     label = np.array(interaction,dtype=np.float32)

    #     atom_feature = torch.FloatTensor(atom_feature)
    #     adj = torch.FloatTensor(adj)
    #     protein = torch.FloatTensor(protein_embedding)
    #     label = torch.LongTensor(label)

    #     compounds.append(atom_feature)
    #     adjacencies.append(adj)
    #     proteins.append(protein)
    #     interactions.append(label)
    # dataset_train_and_vaild = list(zip(compounds, adjacencies, proteins, interactions))
    # compounds, adjacencies,proteins,interactions = [], [], [], []
    # for no, data in tqdm(enumerate(dataset_test),total=len(dataset_test)):
    #     smiles, sequence, interaction = data.strip().split(" ")

    #     atom_feature, adj = mol_features(smiles)
    #     protein_embedding = get_protein_embedding(vec_model, seq_to_kmers(sequence))
    #     label = np.array(interaction,dtype=np.float32)

    #     atom_feature = torch.FloatTensor(atom_feature)
    #     adj = torch.FloatTensor(adj)
    #     protein = torch.FloatTensor(protein_embedding)
    #     label = torch.LongTensor(label)

    #     compounds.append(atom_feature)
    #     adjacencies.append(adj)
    #     proteins.append(protein)
    #     interactions.append(label)
    # dataset_test = list(zip(compounds, adjacencies, proteins, interactions))
    # print('The preprocess of ' + DATASET + ' dataset has finished!')

    K_Fold = 5
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(1,K_Fold+1):
        print('*'*25 + f'No.{i_fold} fold' + '*'*25)
        """ create model ,trainer and tester """
        protein_dim = 100
        atom_dim = 34
        hid_dim = 64
        n_layers = 3
        n_heads = 8
        pf_dim = 256
        dropout = 0.1
        batch = 16
        lr = 1e-4
        weight_decay = 1e-4
        iteration = 200
        kernel_size = 9

        dataset_train, dataset_val = get_kfold_data(i_fold,dataset_train_and_vaild,K_Fold)
        dataset_train = CustomDataSet(dataset_train)
        dataset_val = CustomDataSet(dataset_val)
        dataset_test = CustomDataSet(dataset_test)
        collater_func = collater()
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch, shuffle=True,collate_fn=collater_func,num_workers=6,drop_last=True)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch, shuffle=True,collate_fn=collater_func,num_workers=6,drop_last=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch, shuffle=True,collate_fn=collater_func,num_workers=6,drop_last=True)

        encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
        model = Predictor(encoder, decoder, device)
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
        model.to(device)
        trainer = Trainer(model, lr, weight_decay)
        valer = Valer(model)

        """Output files."""
        file_logs = f'./{DATASET}/{i_fold}/log.txt'
        file_model = f'./{DATASET}/{i_fold}/BestModel'
        AUC = ('Epoch\tTime(sec)\tLoss_train\tAccuracy\tPrecision\tReacll\tAUC\tPRC')
        with open(file_logs, 'w') as f:
            f.write(AUC + '\n')

        """Start training."""
        print('Training...')
        print(AUC)
        scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
        max_Accuracy = 0
        for epoch in range(1, iteration + 1):
            start = timeit.default_timer()
            loss_train = trainer.train(dataloader_train, device)
            Accuracy, Precision, Reacll, AUC, PRC = valer.test(dataloader_val)
            end = timeit.default_timer()
            time = end - start

            AUCs = [epoch, time, loss_train, Accuracy, Precision, Reacll, AUC, PRC]
            scheduler.step()
            valer.save_Acc(AUCs, file_logs)
            if Accuracy > max_Accuracy:
                valer.save_model(model, file_model)
                max_Accuracy = Accuracy
            print('\t'.join(map(str, AUCs)))

        print('Testing...')
        tester = Tester(model,file_model)
        test_Accuracy, test_Precision, test_Reacll, test_AUC, test_PRC = tester.test(dataloader_test)
        print('Test Result:')
        print('Acc: {:.3f} Precision: {:.3f} Recall: {:.3f} AUC: {:.3f} PRC: {:.3f}'.format(
            test_Accuracy,test_Precision,test_Reacll,test_AUC,test_PRC))
        AUC_List_stable.append(test_AUC)
        Accuracy_List_stable.append(test_Accuracy)
        AUPR_List_stable.append(test_PRC)
        Recall_List_stable.append(test_Reacll)
        Precision_List_stable.append(test_Precision)

    show_result(DATASET,
            Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
            AUC_List_stable, AUPR_List_stable)