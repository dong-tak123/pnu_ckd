### train.py 에서는 args.grouped = 0과 1만 다룬다.
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

# from tab_transformer_pytorch import TabTransformer


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

def test(model, criterion, test_loader, logger, args):
    model.eval()
    with torch.no_grad():
        total_label = []
        total_pred =[]
        total_loss = 0
        for test_input, test_label in test_loader:
            test_input = test_input.cuda()
            test_label = test_label.cuda()
            output_test = model(test_input)
            
            if args.focal:
                loss_test = cal_focal_loss(args, output_test, test_label)
            else:
                loss_test = criterion(output_test.squeeze(), test_label)
            total_loss += loss_test.item()

            total_label.extend(test_label.cpu().detach().numpy())
            total_pred.extend(output_test.cpu().detach().numpy())
            
        cm, recall, precision, acc_test, roc_test = metric_test(np.array(total_label), np.array(total_pred))
        msg = "Test set results ==> loss= {:.4f}, recall= {:.4f}, precision= {:.4f}, accuracy= {:.4f}, AUC= {:.4f}".format(
            total_loss/len(total_label), recall.item(), precision.item(), acc_test.item(), roc_test.item()
        )
        logger.info(msg)
        logger.info(cm)

def main(args):
    # seed 및 cuda device 지정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    hiddens = [(64, 64), (64, 32), (64, 16), (64, 8), (32, 32),
               (32, 16), (32, 8), (16, 16), (16, 8), (8, 8)]
    # hiddens = [(8, 4), (4, 4)]
    
    ### original, undersampling, oversampling checking
    # (args.undersampling, args.oversampling)
    # settings = {"original" : (False, False), "undersampling" : (True, False), "oversampling" : (False, True)}
    settings = {"undersampling" : (True, False)}
    
    ### log file 4개 만들어짐.
    now = datetime.now()
    experimentID = now.strftime("%b%d-%H%M%S")
    log_path = "./logs/{}/".format('MLP')
    logger = get_logger(save_dir=log_path, fname='{}-{}.log'.format(experimentID, args.csv_filename))
    
    """
    Basic & Food 
    1. 0922_basic_food_adjusted_sum / 2. 0922_basic_food_adjusted_mean / 3. 0922_basic_food_sum / 4. 0922_basic_food_mean
    
    Basic
    1. 0922_basic_only
    
    Food
    1. 0922_food_adjusted_sum_only / 2. 0922_food_adjusted_mean_only / 3. 0922_food_sum_only / 4. 0922_food_mean_only
    """
    
    for name, setting in settings.items():
        """
        # 1. original -> 2. undersampling -> 3. oversampling
        # """
        args.undersampling = setting[0]
        args.oversampling = setting[1]
    
        # # 기본으로 under sampling, args.oversampling이 True이면 oversampling하도록 하고 싶다.
        train_loader, valid_loader, test_loader, input_dim = load_ckd_after95(args=args, logger=logger)
        
        logger.info(f"Dataset : {args.csv_filename} , Augment : {name}")     # 3번 출력
        logger.info(args)

        for (hidden1, hidden2) in hiddens:
            model = MLP(input_size=input_dim, hidden_size1 = hidden1, hidden_size2=hidden2, output_size=1)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            if args.cuda:
                model.cuda(args.devices[0])

            logger.info(f"Oversamplilng : {args.oversampling}, Undersampling : {args.undersampling}")
            logger.info(f"Model archi : {model.fc1.in_features} - {model.fc2.in_features} - {model.fc3.in_features} - {model.fc3.out_features}")
            
            # Train model
            t_total = time.time()
            for epoch in range(args.epochs):
                train(model, optimizer, criterion, epoch, train_loader, valid_loader, logger, args)
            logger.info("=====Optimization Finished!=====")
            logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

            test(model, criterion, test_loader, logger, args)


if __name__ == "__main__":
    main(args)