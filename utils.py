import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix

def prepare_folders(args):
    folders_util = [args.data_path, args.root_model, os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f'Creating folder: {folder}')
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    filename = f'{args.root_model}/{args.store_name}/ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        print('Save Best checkpoint \n')
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def save_checkpoint_repeat(args, state, is_best, num_repeat):
    filename = f'{args.root_model}/{args.store_name}/repeat{num_repeat}_ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        print('Save Best checkpoint \n')
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    # Cifar_Step1, Cifar_Step 2, Cifar_Step3
    # SVHN_Step1, SVHN_Step2, SVHN_Step3
    if epoch <= 5:
        lr = 0.1 * epoch / 5
    elif 5 < epoch <= 50:
        lr = 0.1  # 1e-1
    elif 80 <= epoch < 90:
        lr = 0.001  # 1e-3
    elif epoch >= 90:
        lr = 0.0001  # 1e-4
    else:
        lr = 0.01  # 1e-2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

