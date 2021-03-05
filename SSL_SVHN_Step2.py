import random
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from Function_SSL_SVHN import *
from utils import *


def setting_args():
    # Setting arguments
    args = define_argument().parse_args()

    # args.data_path = 'D:/ZZZ_img_SVHN/'
    # args.root_model = 'D:/ZZZ_img_SVHN/checkpoint/'
    # args.output_path = args.data_path
    # args.imb_factor = 1 / 50
    # args.imb_factor_unlabel = 1 / 1
    #
    # # args.exp_str_semi = 'True';
    # # args.exp_str_semi = 'Softmax'
    # args.exp_str_semi = 'Softmax_Ths99'
    #
    # # args.evaluate = 'e'

    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type,
                                str(args.imb_factor), str(args.imb_factor_unlabel), 'semi', args.exp_str_semi])
    args.resume = args.root_model + args.store_name + '/ckpt.best.pth.tar'
    print('Save Folder: ', args.root_model + args.store_name)
    print('Save model: ', args.resume)

    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, which can slow down training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')

    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.',
              f'Use GPU: {args.gpu} for training')
    return args

def main_worker():
    # Setting arguments
    args = setting_args()

    unlabeled_file = ''
    isThreshold = False
    if args.exp_str_semi == 'True':
        unlabeled_file = 'svhn_true_labeling.pickle'
        args.output_filename = args.dataset + '_pseudo_labeling_semi_True.pickle'
        isThreshold = False
    elif args.exp_str_semi == 'Softmax':
        unlabeled_file = args.dataset + '_pseudo_labeling_ssl_pretrain.pickle'
        args.output_filename = args.dataset + '_pseudo_labeling_semi_softmax.pickle'
        isThreshold = True
    elif args.exp_str_semi == 'Softmax_Ths99':
        unlabeled_file = args.dataset + '_pseudo_labeling_semi_softmax.pickle'
        args.output_filename = args.dataset + '_pseudo_labeling_semi_softmax_Ths99.pickle'
        isThreshold = True

    best_acc1 = 0
    best_err1 = 0
    train_loader, unlb_loader, val_loader = load_dataset_semi(args, unlabeled_file=unlabeled_file)  # Load Dataset
    model, optimizer, criterion = create_model(args)  # Create Model

    #===== evaluate only (Using Best Models)
    if args.evaluate:
        print('-------- Evaluate --------')
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Checkpoint '{args.resume}' loaded, testing...")
        _, err1, loss1 = validate(val_loader, model, nn.CrossEntropyLoss(), 0, args)
        err1 = err1.detach().cpu().numpy()

        print('Test Datasets: ')
        print('epoch: ', checkpoint['epoch'], ',   Loss: ', loss1, ',   errTop1: ', err1, ',   LR: ', checkpoint['lr'])

        change_pseudo(unlb_loader, model, args, is_threshold=isThreshold)

        print('Test dataset Confusion Matrix')
        validate_confusion_matrix(val_loader, model)
        return

    #===== Start Training
    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epochs + 1):
        epochLR = adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1, err1, _ = validate(val_loader, model, criterion, epoch, args)

        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_err1 = err1

        output_best = 'Best Prec1# %.3f,   Err1# %.3f' % (best_acc1, best_err1)
        print(output_best)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr': epochLR,
        }, is_best)

        if not is_best:
            print('\n')

if __name__ == '__main__':
    main_worker()


