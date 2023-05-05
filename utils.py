from __future__ import division, print_function, absolute_import

import os
import logging
import numpy as np
import torch

class CustomLogger:
    def __init__(self, args):
        args.save = f"{args.save}_{args.seed}"
        self.mode = args.mode
        self.save_root = args.save
        self.log_freq = args.log_freq

        if self.mode == "train":
            os.makedirs(self.save_root, exist_ok=True)
            filename = os.path.join(self.save_root, f"log_{args.seed}.log")
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s.%(msecs)03d - %(message)s",
                datefmt="%b-%d %H:%M:%S",
                filename=filename,
                filemode="w",
            )
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("").addHandler(console)

            logging.info(f"Logger created at {filename}")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s.%(msecs)03d - %(message)s",
                datefmt="%b-%d %H:%M:%S",
            )

        logging.info(f"Torch random seed: {args.seed}")
        self.reset()


    def reset(self):
        if self.mode == "train":
            self.stats = {
                "train": {"loss": [], "acc": []},
                "eval": {"loss": [], "acc": []},
            }
        else:
            self.stats = {"eval": {"loss": [], "acc": []}}

        
    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['eps'] % self.log_freq == 0 and kwargs['eps'] != 0:
                loss_mean = np.mean(self.stats['train']['loss'])
                acc_mean = np.mean(self.stats['train']['acc'])
                self.log_info(
                    f"[{kwargs['eps']:5d}/{kwargs['totaleps']:5d}] loss: {kwargs['loss']:6.4f} ({loss_mean:6.4f}), acc: {kwargs['acc']:6.3f}% ({acc_mean:6.3f}%)"
                    )

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'evaldone':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])
            self.log_info(
                f"[{kwargs['eps']:5d}] Eval ({kwargs['totaleps']:3d} episode) :: loss: {loss_mean:6.4f} (std) {loss_std:6.4f}, acc: {acc_mean:6.3f} (std) {acc_std:5.3f}"
                )

            self.reset()
            return acc_mean

        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))
    
    
    def log_info(self, strout):
        logging.info(strout)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / target.size(0)).item())
        return res[0] if len(res) == 1 else res



def save_ckpt(episode, metalearner, optim, save):
    os.makedirs(os.path.join(save, 'ckpts'), exist_ok=True)
    torch.save({
        'episode': episode,
        'metalearner': metalearner.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(save, 'ckpts', f'meta-learner-{episode}.pth.tar'))


def resume_ckpt(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim

def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(x.dtype)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), dim=1)
