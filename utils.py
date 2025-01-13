import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import logging
import os

def create_save_dir(save_dir: str): 
    if os.path.isdir(save_dir):
        return save_dir
    os.makedirs(save_dir, exist_ok= True)
    return save_dir

def get_logger(logger_name:str='test', save_dir:str='./', fname:str='run.log'):
      # 로그 생성
    logger = logging.getLogger(name=logger_name)
      # 로그 출력 기준 설정
    logger.setLevel(logging.INFO)
    # 로그 출력 형식
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    save_dir = create_save_dir(save_dir)
    file_handler = logging.FileHandler(filename= os.path.join(save_dir, fname))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

def normalizing(X_train, X_valid, X_test):
    # pd.Dataframe -> pd.Dataframe
    wei_train_scaler = StandardScaler()
    # wei_train_scaler =MinMaxScaler()
    X_train_scaled = wei_train_scaler.fit_transform(X_train)
    X_valid_scaled = wei_train_scaler.transform(X_valid)
    X_test_scaled = wei_train_scaler.transform(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled

def tsne_reduction(args, X_train, X_valid, X_test):
    if args.tsne_dim <4:
        tsne = TSNE(n_components=args.tsne_dim, random_state=args.seed)
    else:
        tsne = TSNE(n_components=args.tsne_dim, random_state=args.seed, method='exact')
    # breakpoint()
    X_train_tsne = tsne.fit_transform(X_train.values)
    X_valid_tsne = tsne.fit_transform(X_valid.values)
    X_test_tsne = tsne.fit_transform(X_test.values)
    return X_train_tsne, X_valid_tsne, X_test_tsne

def pca(args, X_train, X_valid, X_test):
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    
    # breakpoint()
    X_train_tsne = pca.fit_transform(X_train.values)
    X_valid_tsne = pca.transform(X_valid.values)
    X_test_tsne = pca.transform(X_test.values)
    return X_train_tsne, X_valid_tsne, X_test_tsne

# test에 나머지 control sample 추가해서 idx 만 반환
def divide_testset(unbalanced_data, ratio, args):
    # train에서 ckd, control index 확인
    total_idx = unbalanced_data.index
    ckd_idx = unbalanced_data[unbalanced_data['onset_tight'] == 1].index        # 실제 ckd
    control_idx = unbalanced_data[unbalanced_data['onset_tight'] == 0].index    # 실제 control
    # print(control_idx, ckd_idx)

    # ckd 갯수와 동일하게 control idx sampling
    rng = np.random.default_rng(1109)   # always same test set
    sampled_ckd_idx = pd.Index(rng.choice(ckd_idx, size=int(len(ckd_idx)*ratio), replace=False))
    sampled_control_idx = pd.Index(rng.choice(control_idx, size=len(sampled_ckd_idx), replace=False)) # test_ckd 갯수와 동일하게 sampling
    
    test_idx = sampled_ckd_idx.append(sampled_control_idx)
    train_idx = total_idx.difference(test_idx)

    # return 실제 ckd, 실제 ckd 갯수와 동일한 갯수의 subject, control_idx - ckd_idx
    return unbalanced_data.loc[train_idx], unbalanced_data.loc[test_idx]

# test에 나머지 control sample 추가해서 idx 만 반환
def _under_sampling_idx(unbalanced_data, args):
    # train에서 ckd, control index 확인
    ckd_idx = unbalanced_data[unbalanced_data['onset_tight'] == 1].index        # 실제 ckd
    control_idx = unbalanced_data[unbalanced_data['onset_tight'] == 0].index    # 실제 control

    # ckd 갯수와 동일하게 control idx sampling
    """
    Control을 CKD와 어떤 비율로 뽑을지.
    """
    rng = np.random.default_rng(seed=args.seed) 
    sampled_control_idx = pd.Index(rng.choice(control_idx, size= len(ckd_idx), replace=False)) # ckd 갯수와 동일하게 sampling
    not_sampled_control_idx = control_idx.difference(sampled_control_idx)

    assert set(sampled_control_idx).issubset(set(control_idx))
    
    balanced_idx = sampled_control_idx.append(ckd_idx)

    # return 실제 ckd, 실제 ckd 갯수와 동일한 갯수의 subject, control_idx - ckd_idx
    return ckd_idx, sampled_control_idx, not_sampled_control_idx, balanced_idx

def undersampling(unbalanced_data, args):
    a, b, c, d = _under_sampling_idx(unbalanced_data, args)
    under_sampled_data = unbalanced_data.loc[d]
    X_undersampled = under_sampled_data.drop(['RID', 'onset_tight'], axis=1)
    y_undersampled = under_sampled_data['onset_tight']
    return X_undersampled, y_undersampled

def oversampling(unbalanced_dataframe, args=None):
    smote = SMOTE(random_state=args.seed)
    temp = unbalanced_dataframe.drop(['RID'], axis=1)
    X_train, y_train = smote.fit_resample(temp, temp['onset_tight'])

    # X_train에는 RID, onset_3 없음.
    return X_train.drop(['onset_tight'], axis=1), y_train

def load_ckd_after95_CV(path='./data/0922_data/', args=None, logger=None):
    """
    CV 할 때는 X_train_scaled, y_train 만 넘겨주면 됨.
    """
    # return tensor
    print('Loading {} dataset...'.format(args.csv_filename))
    
    wei = pd.read_csv(f"{path}{args.csv_filename}.csv")
    
    # wei_train, wei_test = divide_testset(wei, ratio=0.1, args=args)
    assert ~(args.undersampling and args.oversampling)

    if args.undersampling:
        X, y = undersampling(wei, args=args)
    elif args.oversampling:
        X, y = oversampling(wei, args=args)
    else:
        X = wei.drop(['RID', 'onset_tight'], axis=1)
        y = wei['onset_tight']

    X_train, y_train = X, y

    input_dim = X_train.shape[1]
    X_train_scaled, X_train_scaled, X_train_scaled = normalizing(X_train, X_train, X_train)

    print(f"dataset 크기 : {X_train_scaled.shape, y_train.shape}")
    msg = "dataset 크기 : {}, {}".format(
            X_train_scaled.shape, y_train.shape)
    logger.info(msg)
    
    X_train_scaled = torch.tensor(X_train_scaled, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32)

    if args.cuda:
        X_train_scaled = X_train_scaled.cuda()
        y_train = y_train.cuda()

    return X_train_scaled, y_train, input_dim

def load_ckd_after95(path='./data/0922_data/', args=None, logger=None):
    # return tensor
    print('Loading {} dataset...'.format(args.csv_filename))
    
    wei = pd.read_csv(f"{path}{args.csv_filename}.csv")
    # wei = wei.drop(['onset_tight', 'duration'], axis=1)       # for data with duration
    # wei = wei.drop(['onset_tight'], axis=1)       # for data without duartion
    
    wei_train, wei_test = divide_testset(wei, ratio=0.1, args=args)
    assert ~(args.undersampling and args.oversampling)

    if args.undersampling:
        X, y = undersampling(wei_train)
    elif args.oversampling:
        X, y = oversampling(wei_train, args=args)
    else:
        X = wei_train.drop(['RID', 'onset_tight'], axis=1)
        y = wei_train['onset_tight']

    # test는 이미 다 떼놓은 상태
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=args.seed)
    
    X_test = wei_test.drop(['RID', 'onset_tight'], axis=1)
    y_test = wei_test['onset_tight']


    input_dim = X_train.shape[1]
    X_train_scaled, X_valid_scaled, X_test_scaled = normalizing(X_train, X_valid, X_test)

    print(f"dataset 크기 : {X_train_scaled.shape, X_valid_scaled.shape, X_test_scaled.shape}")
    msg = "dataset 크기 : {}, {}, {}".format(
            X_train_scaled.shape, X_valid_scaled.shape, X_test_scaled.shape)
    logger.info(msg)
    
    X_train_scaled = torch.tensor(X_train_scaled, dtype = torch.float32)
    X_valid_scaled = torch.tensor(X_valid_scaled,dtype = torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype = torch.float32)

    if args.cuda:
        X_train_scaled = X_train_scaled.cuda()
        X_valid_scaled = X_valid_scaled.cuda()
        X_test_scaled = X_test_scaled.cuda()
        y_train = y_train.cuda()
        y_valid = y_valid.cuda()
        y_test = y_test.cuda()

    # 데이터로더 생성
    train_dataset = TensorDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(X_valid_scaled, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # breakpoint()
    return train_loader, valid_loader, test_loader, input_dim

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def metric(y_test, y_pred): # label, pred
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred.flatten()
    y_pred = np.where(y_pred > 0.5, 1, 0)

    cm = confusion_matrix(y_test, y_pred)

    if y_test.sum() == 0:
        tn = (y_pred == 0).sum()        # pred를 0으로 하면 맞춘거.
        fp = (y_pred == 1).sum()        # pred를 1로 하면 false positive
        fn, tp = 0, 0
    else:
        tn, fn, tp, fp  = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
    
    try:    # batch에 negative sample 뿐이면 recall에 zerodivision error 발생
        recall = tp / (fn + tp)
    except:
        recall = -1.
    precision = tp / (fp + tp)
    acc = (tp + tn) / (tn + fn + tp+fp)
    # print("Recall : {:04d}".format(recall), "Precision : {:04d}".format(precision), "Acc : {:04d}".format(acc))
    return np.round(recall,4), np.round(precision,4), np.round(acc,4)

def metric_test(y_test, y_pred): # label, pred
    y_pred_prob = y_pred.copy()
    y_pred = np.where(y_pred > 0.5, 1, 0)
    # breakpoint()
    
    cm = confusion_matrix(y_test, y_pred)

    # print(cm)
    if y_test.sum() == 0:
        tn = (y_pred == 0).sum()        # pred를 0으로 하면 맞춘거.
        fp = (y_pred == 1).sum()        # pred를 1로 하면 false positive
        fn, tp = 0, 0
    else:
        tn, fn, tp, fp  = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
    
    try:    # batch에 negative sample 뿐이면 recall에 zerodivision error 발생
        recall = tp / (fn + tp)
    except:
        recall = -1.
    precision = tp / (fp + tp)
    acc = (tp + tn) / (tn + fn + tp+fp)
    auc = roc_auc_score(y_test, y_pred_prob)
    # print("Recall : {:04d}".format(recall), "Precision : {:04d}".format(precision), "Acc : {:04d}".format(acc))
    return cm, np.round(recall,4), np.round(precision,4), np.round(acc,4), np.round(auc, 4)