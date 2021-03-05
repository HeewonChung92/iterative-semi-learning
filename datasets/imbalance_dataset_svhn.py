import os
import pickle
import numpy as np
import torchvision.datasets as datasets


class ImbalanceDataset_SVHN(datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_seed=0, split='train', transform=None, download=False, target_transform=None):
        super(ImbalanceDataset_SVHN, self).__init__(root, split, transform,  target_transform, download)
        np.random.seed(rand_seed)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.data) / cls_num
        img_max = 1000
        # _, counts_class = np.unique(self.labels, return_counts=True)
        # img_max = np.max(counts_class)

        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)

        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        classes = np.concatenate([classes[1:], classes[:1]], axis=0)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0] # ground truth is only used to select samples
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        return self.img_num_list

def UnlabeledDatasetWithPseudoLabel_SVHN(root, unlabeled_file=None, transform=None):
    # unlabeled dataset with Predict
    unlabeled_pseudo = os.path.join(root, unlabeled_file)  # pseudo-label using model trained on imbalanced data
    print("Loading pseudo labels from %s" % unlabeled_pseudo)
    with open(unlabeled_pseudo, 'rb') as f:
        aux_pseudo = pickle.load(f)
    aux_pseudo = aux_pseudo['extrapolated_targets']

    unlabeled_data = datasets.SVHN(root, split='extra', download=True, transform=transform)
    unlabeled_data.labels = aux_pseudo
    return unlabeled_data

class ImbalanceDataset_UnlabeledDatasetWithPseudoLabel_SVHN(datasets.SVHN):
    cls_num = 10
    unlabel_size_factor = 5

    def __init__(self, root, img_num_list, unlabel_imb_factor=1, unlabeled_file=None, rand_seed=0, split='extra', transform=None, download=False, target_transform=None):
        super(ImbalanceDataset_UnlabeledDatasetWithPseudoLabel_SVHN, self).__init__(root, split, transform, target_transform, download)
        # unlabeled
        self.unlabeled_pseudo = os.path.join(root, unlabeled_file)  # pseudo-label using model trained on imbalanced data
        self.unlabel_imb_factor = unlabel_imb_factor

        np.random.seed(rand_seed)
        self.img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(self.img_num_list_unlabeled)

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls_unlabeled):
        aux_data = self.data
        aux_true = self.labels

        # unlabeled dataset with Predict
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_pseudo = pickle.load(f)
        aux_pseudo = aux_pseudo['extrapolated_targets']

        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)

        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        classes = np.concatenate([classes[1:], classes[:1]], axis=0)

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            idx = np.where(aux_true == the_class)[0] # ground truth is only used to select samples
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(aux_data[selec_idx, ...])
            new_targets.extend(aux_pseudo[selec_idx])   # append pseudo-label (Predict)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        return self.img_num_list_unlabeled

class SemiSupervisedImbalance_SVHN(datasets.SVHN):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, unlabel_imb_factor=1, unlabeled_file=None,
                 rand_seed=0, split='train', transform=None, target_transform=None, download=False):
        super(SemiSupervisedImbalance_SVHN, self).__init__(root, split, transform, target_transform, download)

        train_dataset = ImbalanceDataset_SVHN(
            root=root, imb_type=imb_type, imb_factor=imb_factor,
            rand_seed=rand_seed, split='train', download=download, transform=transform)

        unlb_dataset = ImbalanceDataset_UnlabeledDatasetWithPseudoLabel_SVHN(
            root=root, img_num_list=train_dataset.img_num_list,
            unlabel_imb_factor=unlabel_imb_factor, unlabeled_file=unlabeled_file,
            rand_seed=rand_seed, split='extra', transform=transform)

        self.gen_imbalanced_data(train_dataset, unlb_dataset)

    def gen_imbalanced_data(self, train_data, unlb_data):
        new_data = np.concatenate([train_data.data, unlb_data.data])
        new_targets = np.concatenate([train_data.labels, unlb_data.labels])
        self.data = new_data
        self.labels = new_targets

