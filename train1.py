import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate as collate
import argparse
import os
import numpy as np
from maml import Meta
from models import get_cnn
from utils.data import TaskLoader              
from metann import Learner
import learn2learn as l2l
from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_model(model):
    """ Parameter averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def run(rank, size, args):
    """ Distributed Synchronous SGD Example """

    device = torch.device(args.device)

    config = [
        ('conv2d', [1, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 2]),
        ('relu', [True]),
        ('bn2d', [64]),
        ('flatten',),
        ('linear', [64, 2]),
    ]

    transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_dataset =ImageFolder(root='./train/train1',transform=transform)
    test_dataset =ImageFolder(root='./test',transform=transform)
    net = get_cnn(config)
    model = Meta(update_lr=args.update_lr, meta_lr=args.meta_lr, update_step=args.update_step,
                 update_step_test=args.update_step_test, learner=Learner(net)).to(device)
    average_model(model)
    optimizer = model.meta_optim

    for epoch in range(args.epoch):
        epoch_loss = 0.0
        average_model(model)
        train_loader = TaskLoader(train_dataset, args.n_way, args.k_shot, args.k_query, 1000,
                                   batch_size=args.task_num // args.world_size)
        for step, data in enumerate(train_loader):
            data = [[x.to(device) for x in collate(a) + collate(b)] for a, b in data]
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            average_gradients(model)
            optimizer.step()
            
            if step % 1 == 0:
                with torch.no_grad(): 
                    accs_all_train = []
                    train1_loader = TaskLoader(train_dataset, args.n_way, args.k_shot, args.k_query, 10,
                                            batch_size=args.task_num // args.world_size)
                    model.eval()
                    for data_train in train1_loader:
                        data_train = [[x.to(device) for x in collate(a) + collate(b)] for a, b in data_train]
                        with model.logging:
                            loss = model(data_train)
                            accs_all_train.append(model.log['corrects'])
                            optimizer.zero_grad()
                accs = np.array(accs_all_train).mean(axis=0).astype(np.float16)
                print('Rank:', dist.get_rank(), ',epoch:', epoch,  ',step:', step,',Train acc:', accs[0][10])
                optimizer.zero_grad()
                model.train()

            if step % 1 == 0:
                with torch.no_grad(): 
                    accs_all_test = []
                    test_loader = TaskLoader(test_dataset, args.n_way, args.k_shot, args.k_query, 10,
                                            batch_size=args.task_num // args.world_size)
                    model.eval()
                    for data_test in test_loader:
                        data_test = [[x.to(device) for x in collate(a) + collate(b)] for a, b in data_test]
                        with model.logging:
                            loss = model(data_test)
                            accs_all_test.append(model.log['corrects'])
                            optimizer.zero_grad()
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Rank:', dist.get_rank(), ',epoch:', epoch,  ',step:', step,',Test acc:', accs[0][10])
                optimizer.zero_grad
                model.train()

def init_processes(rank, size, fn, args, backend='gloo', addr='127.0.0.2', port='29500'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    parser.add_argument('--n_way', type=int, help='n way', default=2)  
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=1) 
    parser.add_argument('--k_query', type=int, help='k shot for query set', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--device', type=str, help='use CPU', default='cuda')
    parser.add_argument('--world_size', type=int, help='world size of parallelism', default=2)   
    parser.add_argument('--rank', type=int, help='rank', default=0)
    parser.add_argument('--addr', type=str, help='master address', default='127.0.0.2')
    parser.add_argument('--port', type=str, help='master port', default='29500')
    args = parser.parse_args()
    print(args)
    size = args.world_size
    rank = args.rank
    init_processes(rank, size, run, args, addr=args.addr, port=args.port)

