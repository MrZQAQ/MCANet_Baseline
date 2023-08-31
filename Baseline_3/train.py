import copy
from datetime import datetime 
from time import strftime,gmtime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, precision_recall_curve
from torch import nn
from torch.autograd import Variable
from torch.utils import data

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis', 'DrugBank', 'KIBA','Davis'],
                    default='', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'

def get_task_mod(task_name):
    if task_name == 'DrugBank':
        return './dataset_mod/DrugBank'
    elif task_name == 'KIBA':
        return './dataset_mod/KIBA'
    elif task_name == 'Davis':
        return './dataset_mod/Davis'

def test(data_generator, model,extra_output=False):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    with tqdm(enumerate(data_generator),total=len(data_generator)) as pbar:
        pbar.set_description('Testing:')
        for i, (d, p, d_mask, p_mask, label) in pbar:
            score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))

            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score))
            loss_fct = torch.nn.BCELoss()

            label_gpu = Variable(torch.from_numpy(np.array(label)).float()).to(device)

            loss = loss_fct(logits, label_gpu)

            loss_accumulate += loss.item()
            count += 1

            pbar.set_postfix({'loss': '{:.6f}'.format(loss_accumulate/count)})

            logits = logits.detach().cpu().numpy()

            y_label.extend(label.flatten().tolist())
            y_pred.extend(logits.flatten().tolist())

    loss = loss_accumulate / count

    fpr, tpr, _ = roc_curve(y_label, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)
    f1 = (2 * precision * recall) / (precision + recall)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    if extra_output :
        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,outputs), loss,accuracy1,recall_score(y_label, y_pred_s),precision_score(y_label, y_pred_s)
    else:
        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,outputs), loss


def main(fold):
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size


    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'--- Fold {fold+1} start ---')
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True}

    # dataFolder = get_task(args.task)
    dataFolder = get_task_mod(args.task)

    df_train = pd.read_csv(dataFolder + f'/train_{fold+1}.csv')
    df_val = pd.read_csv(dataFolder + f'/val_{fold+1}.csv')
    df_test = pd.read_csv(dataFolder + f'/test.csv')

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    with torch.set_grad_enabled(False):
        auc_num, auprc, f1, loss = test(testing_generator, model_max)
        print(f'Fold {fold+1}: Initial Testing AUROC: ' + str(auc_num) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
            f1) + ' , Test loss: ' + str(loss))

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        with tqdm(enumerate(training_generator),total=len(training_generator)) as pbar:
            pbar.set_description('epoch: [{}/{}]'.format(epo+1,args.epochs))
            for i, (d, p, d_mask, p_mask, label) in pbar:
                score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))

                label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))
                if torch.any(torch.isnan(n)):
                    print(label)
                    print(n)

                loss = loss_fct(n, label)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                pbar.set_postfix({'loss': '{:.6f}'.format(loss.item())})

        # every epoch test
        with torch.set_grad_enabled(False):
            auc_num, auprc, f1, loss = test(validation_generator, model)
            if auc_num > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc_num
                # torch.save('best_checkpoint.pth',model_max)
            print(f'Fold {fold+1}: Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc_num) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc_num, auprc, f1, loss, acc, recall, precision = test(testing_generator, model_max,extra_output=True)
            print(
                f'Fold {fold+1}: Testing AUROC: ' + str(auc_num) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss))
    except:
        print(f'Fold {fold+1}: testing failed')
    return model_max,auc_num, auprc, f1, acc, recall, precision


# 
# model_max, loss_history = main()
# 
# 

K_Fold = 5
auc_list=[]
auprc_list=[]
f1_list=[]
acc_list=[]
recall_list=[]
precision_list=[]
totaltime_list=[]


for fold in range(K_Fold):
    s = datetime.now() 
    _, auc_num, auprc, f1, acc, recall, precision = main(fold)
    e = datetime.now() 
    auc_list.append(auc_num)
    auprc_list.append(auprc)
    f1_list.append(f1)
    acc_list.append(acc)
    recall_list.append(recall)
    precision_list.append(precision)
    totaltime = (e-s).seconds
    totaltime_list.append(totaltime)
    print(f'Fold {fold+1} Total Times: {strftime("%H:%M:%S", gmtime(totaltime))}(second)')
print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1 score: {}\nAUC: {}\nPRC: {}\n'.format(
    np.mean(acc_list),np.mean(precision_list),np.mean(recall_list),np.mean(f1_list),np.mean(auc_list),np.mean(auprc_list)
))
print('Average time cost:{}'.format(strftime("%H:%M:%S", gmtime(np.mean(totaltime_list)))))
