from __future__ import division
from __future__ import print_function
import argparse
import time
import random
import argparse
import torch
import torch.optim as optim
import layers
from dataset import full_load_data
from model import *
from layers import *
from utils import *
import wandb
import matplotlib
wandb.login()


parser = argparse.ArgumentParser()
#training settings
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument("--epoch", type=int, default=30, help='Number of epochs to train')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
parser.add_argument("--layer", type=int, default=0, help='numube of layers')
parser.add_argument('--hidden', type=int, default=16, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--device', type=int, default=0, help='device id')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--model', type=str, default="CGCN", help='')


#dataset settings
parser.add_argument('--row_normalized_adj', action='store_true', default=False, help='choose normalization')
parser.add_argument('--no_degree', action='store_false', default=True, help='do not use degree correction (degree correction only used with symmetric normalization)')
parser.add_argument('--no_sign', action='store_false', default=True, help='do not use signed weights')
parser.add_argument('--no_decay', action='store_false', default=True, help='do not use decaying in the residual connection')
parser.add_argument('--use_bn', action='store_true', default=False, help='use batch norm when not using decaying')
parser.add_argument('--use_ln', action='store_true', default=False, help='use layer norm when not using decaying')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = Augular_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    # print_params(model=model)
    return loss_train.item(), acc_train.item()

def validate_step(model,features,labels,adj,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        #loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        loss_val = Augular_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test_step(model, features, labels, adj, idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    # print(model.alpha)
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))

        return loss_test.item(), acc_test.item()


wandb.init(
    project='CGNN',
    config={
        'lr': args.lr,
        'arc': args.model,
        'data': args.data,
        'epochs': args.epoch,
        'weight_decay': args.weight_decay,
    }
)


def print_params(model):
    for name, param in model.named_parameters():
        print(name, end=" gradient:")
        print(param.grad)
        # print("norm: ", np.linalg.norm(param.grad))

    return 0


def train(datastr, splitstr, checkpt_file):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(
        datastr, splitstr, args.row_normalized_adj, model_type=args.model)
    features = features.to(device)
    adj = adj.to(device)
    if args.model == "CGCN":
        adj = adj.to_dense()
        model = CGNN(num_features=features.shape[1],
                     num_layers=args.layer,
                     num_hidden=args.hidden,
                     num_class=num_labels,
                     dropout=args.dropout,
                     labels=labels,
                     num_nodes=features.shape[0], device=device).to(device)
        # print(features.max())
        #         print(features[0,9])
        #         print(features[0,10])
        #        features = pre_norm(features)#.type(torch.complex64)
        features = features.type(torch.complex64)

    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    #     optimizer = optim.Adam([
    #                             {'params':model.params1,'weight_decay':0.},
    #                             {'params':model.params2,'weight_decay':args.weight_decay}
    #     ],lr=args.lr)

    bad_counter = 0
    best = 999999999999999
    train_loss_draw = []
    val_loss_draw = []
    train_acc_draw = []
    val_acc_draw = []
    for epoch in range(args.epoch):

        #         if epoch <=20:
        #             optimizer = optim.Adam(model.parameters(), lr=1,
        #                             weight_decay=args.weight_decay)
        #         else:
        #             optimizer = optim.Adam(model.parameters(), lr=args.lr,
        #                             weight_decay=args.weight_decay)

        loss_tra, acc_tra = train_step(model, optimizer, features, labels, adj, idx_train)
        loss_val, acc_val = validate_step(model, features, labels, adj, idx_val)
        # loss_val, acc_val = 1,50

        train_loss_draw.append(loss_tra)
        train_acc_draw.append(acc_tra)
        val_loss_draw.append(loss_val)
        val_acc_draw.append(acc_val)
        wandb.log({
            "acc_tra": acc_tra, "loss_tra": loss_tra,
            "acc_val": acc_val, "loss_val": loss_val,
        })

        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
        # break

        if bad_counter == args.patience:
            break
    #
    test_res = test_step(model, features, labels, adj, idx_test)
    acc = test_res[1]
    #     draw_fig(train_loss_draw,val_loss_draw,'loss')
    #     draw_fig(train_acc_draw,val_acc_draw,'acc.')

    return acc * 100




cudaid = "cuda:"+str(args.device)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
checkpt_file = 'pretrained/'+"{}_{}_{}".format(args.model, args.data, current_time)+'.pt'
print(cudaid, checkpt_file)
t_total = time.time()
acc_list = []

for i in range(1):
    datastr = args.data
    splitstr = 'splits/' + args.data + '_split_0.6_0.2_' + str(i) + '.npz'
    acc = train(datastr, splitstr, checkpt_file)
    acc_list.append(acc)
    print(i, ": {:.2f}".format(acc_list[-1]))


print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))

