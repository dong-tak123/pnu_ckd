"""
DL : 36 - 64 - 16 - 1  을 Cross validation 하는 코드.
"""
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from utils import accuracy, load_ckd, metric, load_ckd_balance, metric_test
from datetime import datetime
from utils import *
from models import GCN, MLP, SimpleMLP
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--focal', type=bool, default=False,
                    help='Use Focal Loss of Not')
parser.add_argument('--focal_weight', type=float, default=1.5,
                    help='How much to reduce the effects of control samples')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch Size')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1_lambda', type=float, default=0.01,
                    help='coefficient for L1 loss on parameters.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--devices', nargs='+', default=[0])        # added
parser.add_argument('--oversampling', type=bool, default=False, help='oversampling or not')
parser.add_argument('--undersampling', type=bool, default=False, help='undersampling or not')
parser.add_argument('--csv_filename', type=str, default="0922_basic_food_sum", help='name of dataset')
parser.add_argument('--n_splits', type=int, default=10,
                    help='CV splits')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def cal_focal_loss(args, output, train_label):
    y_ckd = train_label
    log_pred_ckd = torch.log(output.squeeze(1))
    y_control = 1-train_label
    log_pred_control = torch.log(1-output.squeeze(1))
    loss_train = torch.sum(-y_ckd*log_pred_ckd - y_control*log_pred_control/args.focal_weight) / args.batch_size
    return loss_train

def train(model, optimizer, criterion, epoch, train_loader, valid_loader, logger, args):
    t = time.time()
    model.train()
    for train_input, train_label in train_loader:
        train_input = train_input.cuda()
        train_label = train_label.cuda()
        optimizer.zero_grad()
        
        # breakpoint()
        output = model(train_input)
        if args.focal:
            loss_train = cal_focal_loss(args, output, train_label)
        else:
            loss_train = criterion(output.squeeze(), train_label)
            
        if args.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())    # l1 norm
            loss_train += args.l1_lambda * l1_norm

        _, _, acc_train = metric(train_label, output)
        loss_train.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for valid_input, valid_label in valid_loader:
            valid_input = valid_input.cuda()
            valid_label = valid_label.cuda()

            output_valid = model(valid_input)
            
            if args.focal:
                loss_val = cal_focal_loss(args, output_valid, valid_label)
            else:
                loss_val = criterion(output_valid.squeeze(), valid_label)
        
            _, _, acc_val = metric(valid_label, output_valid)
    
    msg = "Train results ==> epoch : {:04d}, loss_train: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}, time: {:.4f}s".format(
        epoch+1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), time.time() - t
    )
    logger.info(msg)

def test_CV(model, criterion, X_test, y_test, logger, args):
    model.eval()
    with torch.no_grad():
        # 데이터를 GPU로 이동
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(args.devices[0] if args.cuda else 'cpu')
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(args.devices[0] if args.cuda else 'cpu')
        
        # 모델 예측
        output_test = model(X_test_tensor)
        
        # 손실 함수 계산
        if args.focal:
            loss_test = cal_focal_loss(args, output_test, y_test_tensor)
        else:
            loss_test = criterion(output_test.squeeze(), y_test_tensor)
        
        # 예측 결과 평가 (예측값을 0.5로 이진 분류)
        total_label = y_test_tensor.cpu().detach().numpy()
        total_pred = output_test.cpu().detach().numpy()
        
        # 정확도, 재현율, 정밀도, AUC 등의 메트릭 계산
        cm, recall, precision, acc_test, roc_test = metric_test(total_label, total_pred)
        
        # 결과 로그 출력
        logger.info("Test set results")
        msg = "Test set results ==> loss= {:.4f}, recall= {:.4f}, precision= {:.4f}, accuracy= {:.4f}, AUC= {:.4f}".format(
            loss_test.item(), recall, precision, acc_test, roc_test)
        logger.info(msg)
        logger.info(cm)

        return loss_test.item(), recall, precision, acc_test, roc_test

def cross_validate(model_class, criterion, optimizer_class, args, logger, X_train_scaled, y_train, input_dim, hidden1, hidden2):
    # KFold 객체 생성 (n_splits = 폴드 수)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # 데이터셋에서 입력과 라벨을 분리
    X = X_train_scaled
    y = y_train

    fold_losses, fold_recalls, fold_precisions, fold_accuracies, fold_aucs = [], [], [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f'Fold {fold+1}')
        logger.info(f'Fold {fold+1}')

        # 학습 데이터와 검증 데이터 분리
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        breakpoint()
        logger.info(f"Train : {X_train.shape}, Test : {X_val.shape}")
        logger.info(f"Train : {y_train.shape}, Test : {y_val.shape}")

        # 데이터로더를 각 폴드마다 생성
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

        # 모델 초기화
        model = model_class(input_size=input_dim, hidden_size1=hidden1, hidden_size2=hidden2, output_size=1)
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()

        # 학습
        for epoch in range(args.epochs):
            train(model, optimizer, criterion, epoch, train_loader, valid_loader, logger, args)

        # 검증
        loss_val, recall_val, precision_val, acc_val, roc_val = test_CV(model, criterion, X_val, y_val, logger, args)
        fold_losses.append(loss_val)
        fold_recalls.append(recall_val)
        fold_precisions.append(precision_val)
        fold_accuracies.append(acc_val)
        fold_aucs.append(roc_val)
        
        logger.info(f"Fold {fold+1} Validation results")
        logger.info(f"Loss, Recall, Precision, ACC, ROC:")
        logger.info(f"{loss_val:.4f}, {recall_val:.4f}, {precision_val:.4f}, {acc_val:.4f}, {roc_val:.4f}")
        

    # 전체 폴드의 평균 정확도 출력
    mean_fold_loss, std_fold_loss = np.mean(fold_losses), np.std(fold_losses)
    mean_fold_recall, std_fold_recall = np.mean(fold_recalls), np.std(fold_recalls)
    mean_fold_pricision, std_fold_precision = np.mean(fold_precisions), np.std(fold_precisions)
    mean_fold_accuracy, std_fold_accuracy = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_fold_auc, std_fold_auc = np.mean(fold_aucs), np.std(fold_aucs)
    logger.info('!!! Cross-Validation Mean Accuracy !!!')
    logger.info(f'loss  /  recall  /  precision  /  ACC  /  ROC  ')
    logger.info(f'MEAN :: {mean_fold_loss:.4f}, {mean_fold_recall:.4f}, {mean_fold_pricision:.4f}, {mean_fold_accuracy:.4f}, {mean_fold_auc:.4f}')
    logger.info(f'STD :: {std_fold_loss:.4f}, {std_fold_recall:.4f}, {std_fold_precision:.4f}, {std_fold_accuracy:.4f}, {std_fold_auc:.4f}')
    
    
    
def main(args):
    # seed 및 cuda device 지정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # hiddens = [(64, 16)]
    hiddens = [(64, 64), (64, 32), (64, 16), (64, 8), (64, 4),
               (32, 32), (32, 16), (32, 8), (32, 4),
               (16, 16), (16, 8), (16, 4), (8, 8), (8, 4), (4, 4)]

    # 로그 파일 설정
    now = datetime.now()
    experimentID = now.strftime("%b%d-%H%M%S")
    log_path = "./logs/{}/{}/".format('MLP', 'traintest91_CV')
    logger = get_logger(save_dir=log_path, fname='{}-{}-seed{}-splits{}.log'.format(experimentID, args.csv_filename, args.seed, args.n_splits))
    
    settings = {"undersampling" : (True, False)}
    
    for name, setting in settings.items():
        args.undersampling = setting[0]
        args.oversampling = setting[1]

        # 데이터를 로드
        X_train_scaled, y_train, input_dim = load_ckd_after95_CV(args=args, logger=logger)
        
        logger.info(f"Dataset : {args.csv_filename} , Augment : {name}")
        logger.info(args)

        # 교차 검증을 통해 모델을 학습 및 평가
        for (hidden1, hidden2) in hiddens:
            logger.info(f"Oversamplilng : {args.oversampling}, Undersampling : {args.undersampling}")
            logger.info(f"Training with hidden layers {hidden1} and {hidden2}")
            
            cross_validate(MLP, nn.BCELoss(), optim.Adam, args, logger, X_train_scaled, y_train, input_dim, hidden1, hidden2)


if __name__ == "__main__":
    main(args)