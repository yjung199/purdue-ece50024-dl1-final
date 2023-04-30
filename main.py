from __future__ import division, print_function, absolute_import

import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from utils import *
# from models import *
from dataloader import prepare_data
from metalearner import MetaLearner
from learner import Learner
from utils import *
from tqdm import tqdm


def parse_args():

    # Define command-line arguments
    parser = argparse.ArgumentParser()

    # Mode argument
    parser.add_argument('--mode', choices=['train', 'test'], help='train or test')

    # Hyperparameters
    parser.add_argument('--num_shot', default=1, type=int, help='number of support examples per class for train')
    # parser.add_argument('--n-shot', type=int, help="How many examples per class for training (k, n_support)")
    parser.add_argument('--num_eval', default=15, type=int, help='number of query examples per class for eval')
    # parser.add_argument('--n-eval', type=int, help="How many examples per class for evaluation (n_query)")
    parser.add_argument('--num_class', default=5, type=int, help='number of classes per episode')
    # parser.add_argument('--n-class', type=int, help="How many classes (N, n_way)")
    parser.add_argument('--input_size', default=128, type=int, help='input size of initial meta-learner')
    # parser.add_argument('--input-size', type=int, help="Input size for the first LSTM")
    parser.add_argument('--hidden_size', default=40, type=int, help='hidden size of meta-learner')
    # parser.add_argument('--hidden-size', type=int, help="Hidden size for the first LSTM")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--episode', type=int, help="number of episodes for train")
    parser.add_argument('--episode_val', type=int, help="number of episodes for eval")

    parser.add_argument('--epoch', type=int, help="number of epoch to train for an episode")
    parser.add_argument('--batch_size', type=int, help="batch size when training an episode")
    parser.add_argument('--image_size', type=int, help="size for the image to be resized")
    parser.add_argument('--grad_clip', type=float, help="gradient clipping")
    parser.add_argument('--bn_momentum', type=float, help="batchnorm epsilon")
    parser.add_argument('--bn_eps', type=float, help="batchnorm episode")

    # Paths
    parser.add_argument('--dataset', choices=['miniimagenet'], help="Name of dataset")
    parser.add_argument('--data_root', type=str, help="Location of data")
    parser.add_argument('--resume', type=str, help="Location to pth.tar")
    parser.add_argument('--save', type=str, default='logs', help="Location to logs and ckpts")

    # Others
    parser.add_argument('--cpu', action='store_true', help="use CPU instead of GPU")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for DataLoader")
    parser.add_argument('--pin_mem', type=bool, default=False, help="DataLoader pin_memory")
    parser.add_argument('--log_freq', type=int, default=100, help="logging frequency")
    parser.add_argument('--val_freq', type=int, default=1000, help="validation frequency")
    parser.add_argument('--seed', type=int, help="random seed")

    # args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', default='train', help='train or test')
    # parser.add_argument('--dataset', default='miniimagenet', help='name of dataset')
    # parser.add_argument('--data_root', default='./data', help='path to dataset')
    # parser.add_argument('--save', default='./checkpoints', help='path to save checkpoints')
    # parser.add_argument('--log', default='./logs', help='path to save logs')
    # parser.add_argument('--resume', default='', help='path to checkpoint')
    # parser.add_argument('--episode', default=60000, type=int, help='number of episodes')
    # parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    # parser.add_argument('--bn_eps', default=1e-3, type=float, help='batchnorm epsilon')
    # parser.add_argument('--bn_momentum', default=0.05, type=float, help='batchnorm momentum')
    # parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    
    # parser.add_argument('--image_size', default=28, type=int, help='image size')
    
    
    # parser.add_argument('--seed', default=None, type=int, help='random seed')
    # parser.add_argument('--cpu', action='store_true', help='use cpu')
    # parser.add_argument('--val_freq', default=2000, type=int, help='validation frequency')
    
    return parser
    


def set_device(args):
    if args.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
    return device


def initialize_logger(args):
    logger = GOATLogger(args)
    return logger


def get_data_loaders(args):
    train_loader, val_loader, test_loader = prepare_data(args)
    return train_loader, val_loader, test_loader


def setup_learners_and_metalearner(args, device):
    learner_w_grad = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.num_class).to(device)
    learner_wo_grad = copy.deepcopy(learner_w_grad)
    metalearner = MetaLearner(args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)).to(device)
    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())
    return learner_w_grad, learner_wo_grad, metalearner

def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            x = train_input[i:i+args.batch_size]
            y = train_target[i:i+args.batch_size]

            # get the loss/grad
            learner_w_grad.copy_flat_params(cI)
            output = learner_w_grad(x)
            loss = learner_w_grad.criterion(output, y)
            acc = accuracy(output, y)
            learner_w_grad.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0)

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            #print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

    return cI

def meta_test(episode, eval_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger):
    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        train_input = episode_x[:, :args.num_shot].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [num_class * num_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.num_class), args.num_shot)).to(args.device) # [num_class * num_shot]
        test_input = episode_x[:, args.num_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [num_class * num_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.num_class), args.num_eval)).to(args.device) # [num_class * num_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=episode, totaleps=args.episode_val, phase='evaldone')

def main():
    args, unparsed = parse_args().parse_known_args()
    if len(unparsed) != 0:
        raise RuntimeError("Unknown arguments: {}".format(unparsed))
    
    if args.seed is None:
        args.seed = random.randint(0, 1e3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = set_device(args)
    args.device = device
    logger = initialize_logger(args)
    train_loader, val_loader, test_loader = get_data_loaders(args)
    learner_w_grad, learner_wo_grad, metalearner = setup_learners_and_metalearner(args, device)

    # Set optimizer and loss function
    optimizer = torch.optim.Adam(metalearner.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        logger.loginfo("Resuming from checkpoint: {}".format(args.resume))
        last_eps, metalearner, optimizer = resume_ckpt(metalearner, optimizer, args.resume, args.device)
        logger.loginfo("Resumed from checkpoint: {}".format(args.resume))
    
    # Train metalearner
    best_acc = 0.0
    for episode, (eps_x, eps_y) in enumerate(train_loader):
        train_input = eps_x[:, :args.num_shot].reshape(-1, *eps_x.shape[-3:]).to(args.device) # [num_class * num_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.num_class), args.num_shot)).to(args.device) # [num_class * num_shot]
        test_input = eps_x[:, args.num_shot:].reshape(-1, *eps_x.shape[-3:]).to(args.device) # [num_class * num_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.num_class), args.num_eval)).to(args.device) # [num_class * num_eval]

        # Train learner with meta learner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.train()
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        optimizer.step()

        logger.batch_info(eps=episode, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if episode % args.val_freq == 0 and episode != 0:
            save_ckpt(episode, metalearner, optimizer, args.save)
            acc = meta_test(episode, val_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")


    logger.loginfo("Done")

if __name__ == '__main__':
    main()
