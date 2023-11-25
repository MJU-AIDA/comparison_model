# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import random
import os

# Set the CUDA_VISIBLE_DEVICES environment variable
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping  
import timeit

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, f1_score


def show_result(DATASET,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
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

def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, f1_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f};f1_score :{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, f1_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(3)
    torch.cuda.empty_cache()
    print(f"Number of available GPUs: {num_gpus}")
else:
    print("CUDA is not available")
    



if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    DATASET = "vec"
    #DATASET = "KIBA"
    # DATASET = "DrugBank"
    # DATASET = "Davis"
    print("Train in " + DATASET)
    
    if DATASET == "vec":
        weight_CE = None
        dir_input = ('./data/{}.txt'.format(DATASET))
        train_dir_input = ('./data/train_{}.txt'.format(DATASET))
        dev_dir_input = ('./data/dev_{}.txt'.format(DATASET))
        test_dir_input = ('./data/test_{}.txt'.format(DATASET))
        
        print("load data")
        with open(train_dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        with open(dev_dir_input, "r") as f:
            dev_data_list = f.read().strip().split('\n')
        with open(test_dir_input, "r") as f:
            test_data_list = f.read().strip().split('\n')
        print("load finished")
        
    train_dataset = train_data_list
    valid_dataset = dev_data_list
    test_dataset = test_data_list
    

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    
    
    train_dataset = CustomDataSet(train_dataset)
    valid_dataset = CustomDataSet(valid_dataset)
    test_dataset = CustomDataSet(test_dataset)
    #TVdataset_len = len(TVdataset)
    #valid_size = int(0.2 * TVdataset_len)
    train_size = len(train_dataset)
    #train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
    train_dataset_load = DataLoader(train_dataset, batch_size= 32, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    valid_dataset_load = DataLoader(valid_dataset, batch_size= 32, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn)
    test_dataset_load = DataLoader(test_dataset, batch_size= 32, shuffle=False, num_workers=0,
                                   collate_fn=collate_fn)


    """ create model"""
    model = AttentionDTI(hp).cuda()
    model.load_state_dict(torch.load("vec/result/valid_best_checkpoint.pth"))
    Loss = nn.CrossEntropyLoss(weight=weight_CE)
    save_path = "vec/"
    note = ""
    writer = SummaryWriter(log_dir=save_path, comment=note)
    
    
    """valid"""
    valid_pbar = tqdm(
        enumerate(
            BackgroundGenerator(valid_dataset_load)),
        total=len(valid_dataset_load))
    # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
    valid_losses_in_epoch = []
    model.eval()
    Y, P, S = [], [], []
    with torch.no_grad():
        for valid_i, valid_data in valid_pbar:
            '''data preparation '''
            valid_compounds, valid_proteins, valid_labels = valid_data

            valid_compounds = valid_compounds.cuda()
            valid_proteins = valid_proteins.cuda()
            valid_labels = valid_labels.cuda()

            valid_scores = model(valid_compounds, valid_proteins)
            valid_loss = Loss(valid_scores, valid_labels)
            valid_labels = valid_labels.to('cpu').data.numpy()
            valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
            valid_predictions = np.argmax(valid_scores, axis=1)
            valid_scores = valid_scores[:, 1]

            valid_losses_in_epoch.append(valid_loss.item())
            Y.extend(valid_labels)
            P.extend(valid_predictions)
            S.extend(valid_scores)
    Precision_dev = precision_score(Y, P)
    Reacll_dev = recall_score(Y, P)
    Accuracy_dev = accuracy_score(Y, P)
    f1_dev = f1_score(Y, P)
    AUC_dev = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC_dev = auc(fpr, tpr)
    valid_loss_a_epoch = np.average(valid_losses_in_epoch)  
    # avg_valid_loss.append(valid_loss)

    epoch_len = len(str(hp.Epoch))

    print_msg = (#f'train_loss: {train_loss_a_epoch:.5f} ' +
                 f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                 f'valid_AUC: {AUC_dev:.5f} ' +
                 f'valid_PRC: {PRC_dev:.5f} ' +
                 f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                 f'valid_Precision: {Precision_dev:.5f} ' +
                 f'valid_Reacll: {Reacll_dev:.5f} ' +
                f'valid_f1_score: {f1_dev:.5f}')

    #writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
    writer.add_scalar('Valid AUC', AUC_dev)
    writer.add_scalar('Valid AUPR', PRC_dev)
    writer.add_scalar('Valid Accuracy', Accuracy_dev)
    writer.add_scalar('Valid Precision', Precision_dev)
    writer.add_scalar('Valid Reacll', Reacll_dev)
    writer.add_scalar('Valid f1_score', f1_dev)
    #writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

    print(print_msg)
    
    
    """test"""
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
    test_losses_in_epoch = []
    model.eval()
    Y, P, S = [], [], []
    with torch.no_grad():
        for test_i, test_data in test_pbar:
            '''data preparation '''
            test_compounds, test_proteins, test_labels = test_data

            test_compounds = test_compounds.cuda()
            test_proteins = test_proteins.cuda()
            test_labels = test_labels.cuda()

            test_scores = model(test_compounds, test_proteins)
            test_loss = Loss(test_scores, test_labels)
            test_labels = test_labels.to('cpu').data.numpy()
            test_scores = F.softmax(test_scores, 1).to('cpu').data.numpy()
            test_predictions = np.argmax(test_scores, axis=1)
            test_scores = test_scores[:, 1]

            test_losses_in_epoch.append(test_loss.item())
            Y.extend(test_labels)
            P.extend(test_predictions)
            S.extend(test_scores)
    Precision_test = precision_score(Y, P)
    Reacll_test = recall_score(Y, P)
    Accuracy_test = accuracy_score(Y, P)
    f1_test = f1_score(Y, P)
    AUC_test = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC_test = auc(fpr, tpr)
    test_loss_a_epoch = np.average(test_losses_in_epoch)  
    # avg_valid_loss.append(valid_loss)

    epoch_len = len(str(hp.Epoch))

    print_msg = (#f'train_loss: {train_loss_a_epoch:.5f} ' +
                 f'test_loss: {test_loss_a_epoch:.5f} ' +
                 f'test_AUC: {AUC_test:.5f} ' +
                 f'test_PRC: {PRC_test:.5f} ' +
                 f'test_Accuracy: {Accuracy_test:.5f} ' +
                 f'test_Precision: {Precision_test:.5f} ' +
                 f'test_Reacll: {Reacll_test:.5f} ' +
                f'test_f1_score: {f1_test:.5f}')

    writer.add_scalar('Test Loss', test_loss_a_epoch)
    writer.add_scalar('Test AUC', AUC_test)
    writer.add_scalar('Test AUPR', PRC_test)
    writer.add_scalar('Test Accuracy', Accuracy_test)
    writer.add_scalar('Test Precision', Precision_test)
    writer.add_scalar('Test Reacll', Reacll_test)
    writer.add_scalar('Test f1_score', f1_test)
    #writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

    print(print_msg)