import random
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from Function_SSL_Cifar import *
from utils import *


arr_repeat_errTop1 = []
arr_repeat_Loss1 = []
arr_repeat_epoch = []

def setting_args():
    # Setting arguments
    args = define_argument().parse_args()

    # args.data_path = 'D:/ZZZ_img_Cifar/'
    # args.root_model = 'D:/ZZZ_img_Cifar/checkpoint/'
    # args.output_path = args.data_path
    # args.imb_factor = 1 / 50
    # args.imb_factor_unlabel = 1 / 1
    # args.exp_str_semi = 'Softmax_Ths99'
    #
    # args.exp_str_class = 'All'
    # # args.exp_str_class = 'Etc'
    # 
    # args.evaluate = 'e'

    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type,
                                str(args.imb_factor), str(args.imb_factor_unlabel), 'semi', args.exp_str_semi, args.exp_str_class])
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

def main_repeat():
    # Setting arguments
    args = setting_args()
    total_repeat = 50

    is_all = True
    if args.exp_str_class == 'All':
        is_all = True
    elif args.exp_str_class == 'Etc':
        is_all = False

    # For Training and Evaluation
    for num_repeat in range(1, total_repeat+1):
        print('===== Repeat: ', num_repeat, ' =====')
        if args.evaluate:
            unlabeled_file = args.dataset + '_pseudo_labeling_semi_softmax_Ths99_' + args.exp_str_class + '_repeat_' + str(num_repeat) + '.pickle'
            args.resume = args.root_model + args.store_name + '/repeat' + str(num_repeat) + '_ckpt.best.pth.tar'
        else:
            if num_repeat == 1:
                unlabeled_file = args.dataset + '_pseudo_labeling_semi_softmax_Ths99.pickle'
                args.resume = args.root_model + args.dataset + '_resnet32_CE_None_exp_' + str(args.imb_factor) + '_' + str(args.imb_factor_unlabel) + '_semi_Softmax_Ths99/ckpt.best.pth.tar'
            else:
                unlabeled_file = args.dataset + '_pseudo_labeling_semi_softmax_Ths99_' + args.exp_str_class + '_repeat_' + str(num_repeat-1) + '.pickle'
                args.resume = args.root_model + args.store_name + '/repeat' + str(num_repeat-1) + '_ckpt.best.pth.tar'
        print('unlabeled file: ', unlabeled_file)
        print('init model: ', args.resume)

        # Start Training or evaluation
        main_worker(args, unlabeled_file, num_repeat, is_all)
        torch.cuda.empty_cache()  # PyTorch thing

    # For Evaluation only
    if args.evaluate:
        for num_repeat in range(1, total_repeat+1):
            print('=== Repeat: ', num_repeat, ', epoch: ', arr_repeat_epoch[num_repeat-1], ', loss: ', arr_repeat_Loss1[num_repeat-1], ',   errTop1: ', arr_repeat_errTop1[num_repeat-1])

def main_worker(args, unlabeled_file, num_repeat, is_all):
    best_acc1 = 0
    best_err1 = 0
    train_loader, unlb_loader, val_loader = load_dataset_semi(args, unlabeled_file=unlabeled_file)   # Load Dataset
    model, optimizer, criterion = create_model(args)   # Create Model

    #===== evaluate only (Using Best Models)
    if args.evaluate:
        print('-------- Evaluate --------')
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Checkpoint '{args.resume}' loaded, testing...")
        _, err1, loss1 = validate(val_loader, model, nn.CrossEntropyLoss(), 0, args)
        validate_confusion_matrix(val_loader, model)

        arr_repeat_errTop1.append(err1.detach().cpu().numpy())
        arr_repeat_Loss1.append(loss1)
        arr_repeat_epoch.append(checkpoint['epoch'])
        return

    #===== Using Previous Model
    if args.resume:
        print('-------- Previous Model Weight&bias --------')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            print('-------------- Initialize to existing model --------------')
        else:
            raise ValueError(f"No checkpoint found at '{args.resume}'")

    #===== Start Training
    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epochs+1):
        epochLR = adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1, err1, _ = validate(val_loader, model, criterion, epoch, args)

        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_err1 = err1

        output_best = 'Best Prec1# %.3f,   Err1# %.3f' % (best_acc1, best_err1)
        print(output_best)

        save_checkpoint_repeat(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr': epochLR,
        }, is_best, num_repeat)

        if not is_best:
            print('\n')

    #===== Finished Training
    print('===== Make Pseudo Label Repeat: ', num_repeat, ' =====')
    torch.cuda.empty_cache()  # PyTorch thing
    args.resume = args.root_model + args.store_name + '/repeat' + str(num_repeat) + '_ckpt.best.pth.tar'
    args.output_filename = args.dataset + '_pseudo_labeling_semi_softmax_Ths99_' + args.exp_str_class + '_repeat_' + str(num_repeat) + '.pickle'
    model, _, _ = create_model(args)  # Create Model
    print('-------- Load Best Model Weight&bias --------')
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    change_pseudo_repeat(unlb_loader, model, args, is_all=is_all)   # Create pseudo label with unlabeled data
    print('\n')

if __name__ == '__main__':
    main_repeat()
