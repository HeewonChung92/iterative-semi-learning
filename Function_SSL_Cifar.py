import argparse
import time
from torch.utils.data import DataLoader

import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms

from datasets.imbalance_dataset_cifar import *
import models
from utils import *
from losses import *

def define_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_path', type=str, default='D:/img_Cifar/')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
    parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'Focal', 'LDAM'])
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--imb_factor_unlabel', default=0.01, type=float, help='imbalance factor for unlabeled data')
    parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'Resample', 'Reweight', 'DRW'])
    parser.add_argument('--rand_seed', default=0, type=int, help='fix random number for data sampling')

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--root_model', type=str, default='D:/img_Cifar/checkpoint/')
    parser.add_argument('--data_filename', default='ti_80M_selected.pickle', type=str)
    parser.add_argument('--output_path', default='D:/img_Cifar/', type=str)
    parser.add_argument('--output_filename', default='pseudo_labeling.pickle', type=str)

    parser.add_argument('--exp_str_semi', default='Softmax', type=str, choices=['Softmax', 'Softmax_Ths99', 'True'])
    parser.add_argument('--exp_str_class', default='All', type=str, choices=['All', 'Etc'])
    return parser

def load_dataset_pretrain(args):
    mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
    std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = ImbalanceDataset_CIFAR10(
        root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
        rand_seed=args.rand_seed, train=True, download=True, transform=transform_train)

    unlb_dataset = UnlabeledDatasetWithPseudoLabel_CIFAR10(root=args.data_path, unlabeled_file='ti_80M_selected.pickle', transform=transform_val)
    val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_val)

    print('--------------------------------------------------------------------------------------------------------')
    aa, counts_train = np.unique(train_dataset.targets, return_counts=True)
    print('Distribution of classes in Original Train data: ', aa, counts_train, sum(counts_train))
    aa, counts_aa = np.unique(unlb_dataset.targets, return_counts=True)
    print('Distribution of classes in Unlabeled Train data: ', aa, counts_aa, sum(counts_aa))
    aa, counts_test = np.unique(val_dataset.targets, return_counts=True)
    print('Distribution of classes in Test: ', aa, counts_test, sum(counts_test))
    print('--------------------------------------------------------------------------------------------------------')

    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    unlb_loader = torch.utils.data.DataLoader(unlb_dataset, batch_size=100, shuffle=False)
    return train_loader, unlb_loader, val_loader

def load_dataset_semi(args, unlabeled_file):
    mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
    std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = SemiSupervisedImbalance_CIFAR10(
        root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
        unlabel_imb_factor=args.imb_factor_unlabel, unlabeled_file=unlabeled_file,
        rand_seed=args.rand_seed, train=True, transform=transform_train
    )

    unlb_dataset = UnlabeledDatasetWithPseudoLabel_CIFAR10(root=args.data_path, unlabeled_file=unlabeled_file, transform=transform_val)
    val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_val)

    print('--------------------------------------------------------------------------------------------------------')
    aa, counts_train = np.unique(train_dataset.targets, return_counts=True)
    print('Distribution of classes in Original Train & Unlabeled Train data: ', aa, counts_train, sum(counts_train))
    aa, counts_aa = np.unique(unlb_dataset.targets, return_counts=True)
    print('Distribution of classes in Unlabeled Train data: ', aa, counts_aa, sum(counts_aa))
    aa, counts_test = np.unique(val_dataset.targets, return_counts=True)
    print('Distribution of classes in Test: ', aa, counts_test, sum(counts_test))

    #===== Remove the classes 10 (this is extra classes)
    idx_stay = np.where(train_dataset.targets < 10)
    train_dataset.data = train_dataset.data[idx_stay, :, :, :].squeeze()
    train_dataset.targets = train_dataset.targets[idx_stay]
    aa, counts_train = np.unique(train_dataset.targets, return_counts=True)
    print('Distribution of classes in Original Train & Unlabeled Train data Remove Etc Classes: ', aa, counts_train, sum(counts_train))
    print('--------------------------------------------------------------------------------------------------------')

    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    unlb_loader = torch.utils.data.DataLoader(unlb_dataset, batch_size=100, shuffle=False)
    return train_loader, unlb_loader, val_loader

def create_model(args):
    print(f"===> Creating model '{args.arch}'")
    num_classes = 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss(weight=None).cuda(args.gpu)  # log_Softmax 뱉을 때,
    return model, optimizer, criterion

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    errTop1 = AverageMeter('Err@1', ':6.2f')
    errTop5 = AverageMeter('Err@5', ':6.2f')

    model.train()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        target = target.cuda()
        target = target.long()

        output = torch.log(model(inputs) + 1e-8)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        errTop1.update(100 - acc1[0], inputs.size(0))
        errTop5.update(100 - acc5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}/{1}][{2}/{3}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Err@1 {errTop1.val:.3f} ({errTop1.avg:.3f})\t'
                      'Err@5 {errTop5.val:.3f} ({errTop5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, errTop1=errTop1, errTop5=errTop5, lr=optimizer.param_groups[-1]['lr']))
            print(output)

def validate(val_loader, model, criterion, epoch, args, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    errTop1 = AverageMeter('Err@1', ':6.2f')
    errTop5 = AverageMeter('Err@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()
            target = target.long()

            output = torch.log(model(inputs) + 1e-8)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            errTop1.update(100 - acc1[0], inputs.size(0))
            errTop5.update(100 - acc5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}][{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec1# {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec5# {top5.val:.3f} ({top5.avg:.3f})\t'
                          'Err1# {errTop1.val:.3f} ({errTop1.avg:.3f})\t'
                          'Err5# {errTop5.val:.3f} ({errTop5.avg:.3f})'.format(
                    epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5, errTop1=errTop1, errTop5=errTop5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec1# {top1.avg:.3f} Prec5# {top5.avg:.3f} Err1# {errTop1.avg:.3f} Err5# {errTop5.avg:.3f} Loss {loss.avg:.5f}'
            .format(flag=flag, top1=top1, top5=top5, errTop1=errTop1, errTop5=errTop5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
    return top1.avg, errTop1.avg, losses.avg

def validate_confusion_matrix(val_loader, model):
    # switch to evaluate mode
    model.eval()
    myPred = []
    myTrue = []
    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            _, pred = torch.max(output, 1)
            myPred.extend(pred.cpu().numpy())
            myTrue.extend(target.cpu().numpy())
    myTrue = np.asarray(myTrue)
    myPred = np.asarray(myPred)

    print('Shape of Test datasets: ', myTrue.shape, myPred.shape)
    from sklearn.metrics import multilabel_confusion_matrix
    aa = multilabel_confusion_matrix(myTrue, myPred)
    print('Confusion matrix of Test datasets: ')
    for ii in range(0, 10):
        print('Confusion class: ', ii+1, ',   => ', aa[ii, 0, 0], aa[ii, 0, 1], aa[ii, 1, 0], aa[ii, 1, 1])

def change_pseudo(unlb_loader, model, args, is_threshold=False):
    print('----- Change pseudo labels in Unlabeled datasets -----')
    out_path = os.path.join(args.output_path, args.output_filename)
    print('Is files: ', os.path.join(args.output_path, args.output_filename), os.path.isfile(out_path))

    if not os.path.isfile(out_path):
        # Running model on unlabeled data
        model.eval()
        predictions = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(unlb_loader):
                preds = model(inputs.cuda())
                zProb, zClass = torch.max(preds, dim=1)

                if is_threshold:
                    zNew = [np.int(zClass[idx].cpu().numpy().squeeze()) if zProb[idx].detach().cpu().numpy() >= 0.99 else 10 for idx in range(0, preds.shape[0])]
                    predictions.append(zNew)
                else:
                    predictions.append(zClass.detach().cpu().numpy())

                if (i + 1) % 500 == 0:
                    print('Done %d/%d' % (i + 1, len(unlb_loader)))

        new_extrapolated_targets = np.concatenate(predictions)
        new_targets = dict(extrapolated_targets=new_extrapolated_targets, prediction_model=args.resume)

        ### Save pickle file (predicted pseudo labels)
        with open(out_path, 'wb') as f:
            pickle.dump(new_targets, f)
        print('Save Pseudo labels')

def change_pseudo_repeat(unlb_loader, model, args, is_all=False):
    print('----- Change pseudo labels in Unlabeled datasets -----')
    out_path = os.path.join(args.output_path, args.output_filename)
    print('Is files: ', os.path.join(args.output_path, args.output_filename), os.path.isfile(out_path))

    if not os.path.isfile(out_path):
        # Running model on unlabeled data
        model.eval()
        predictions = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(unlb_loader):
                preds = model(inputs.cuda())
                zProb, zClass = torch.max(preds, dim=1)

                zNew_all = [np.int(zClass[idx].cpu().numpy().squeeze()) if zProb[idx].detach().cpu().numpy() >= 0.99 else 10 for idx in range(0, preds.shape[0])]
                if is_all:
                    predictions.append(zNew_all)
                else:
                    zNew_etc = [target[idx] if target[idx] < 10 else zNew_all[idx] for idx in range(0, target.shape[0])]
                    predictions.append(zNew_etc)

                if (i + 1) % 500 == 0:
                    print('Done %d/%d' % (i + 1, len(unlb_loader)))

        new_extrapolated_targets = np.concatenate(predictions)
        new_targets = dict(extrapolated_targets=new_extrapolated_targets, prediction_model=args.resume)

        ### Save pickle file (predicted pseudo labels)
        with open(out_path, 'wb') as f:
            pickle.dump(new_targets, f)
        print('Save Pseudo labels')