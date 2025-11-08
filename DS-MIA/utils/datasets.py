import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.sampling import *
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
import random
import numpy as np
from collections import defaultdict


def stratified_split_dataset(dataset, frac=0.5, seed=42):

    label2idxs = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label2idxs[int(label)].append(idx)


    random.seed(seed)
    idxs_a, idxs_b = [], []
    for label, idxs in label2idxs.items():
        random.shuffle(idxs)
        split = int(len(idxs) * frac)
        idxs_a.extend(idxs[:split])
        idxs_b.extend(idxs[split:])


    random.shuffle(idxs_a)
    random.shuffle(idxs_b)
    return idxs_a, idxs_b
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


from typing import List, Tuple

def balance_by_tail_move(idxs_a: List[int], idxs_b: List[int]) -> Tuple[List[int], List[int]]:

    a, b = list(idxs_a), list(idxs_b)
    Na, Nb = len(a), len(b)
    if Na == Nb:
        return a, b

    diff = abs(Na - Nb)
    if diff % 2 != 0:
        raise ValueError(
            f"无法通过移动差值的一半实现相等：|A|={Na}, |B|={Nb}, diff={diff} 为奇数。"
        )

    move_n = diff // 2
    if Na > Nb:
        moved = a[-move_n:]     # 从 A 末尾取
        a = a[:-move_n]
        b += moved              # 加到 B 的末尾
    else:
        moved = b[-move_n:]     # 从 B 末尾取
        b = b[:-move_n]
        a += moved


    assert len(a) == len(b), f"仍不相等：|A|={len(a)}, |B|={len(b)}"
    assert len(set(a).intersection(b)) == 0, "A/B 索引发生重合（理论上不应出现）"
    return a, b


def get_data(mode, dataset, data_root, iid, num_users,data_aug, noniid_beta):

    ds = dataset
    # -------------------- CIFAR-10  --------------------
    if ds == 'cifar10':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                         std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.8),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = torchvision.datasets.CIFAR10(data_root,
           train=True,
           download=True,
           transform=transform_train
        )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(
            data_root,
            train=False,
            download=False,
            transform=transform_test
        )
        train_set_mia = train_set
        test_set_mia = test_set
        idxs_a, idxs_b = stratified_split_dataset(train_set, frac=0.5, seed=42)

        subset_shadow = DatasetSplit(train_set, idxs_a)
        subset_target = DatasetSplit(train_set, idxs_b)

        print(f"Subset shadow: {len(subset_shadow)}")
        print(f"Subset target: {len(subset_target)}")


        if mode == True:  # mode=self.trainShadowModel
            train_set = subset_shadow
            print("We are currently training a shadow model using a shadow dataset")
        else:
            train_set = subset_target
            print("We are currently training the target model using the target dataset")
    # -------------------- CIFAR-100  --------------------
    if ds == 'cifar100':
        if data_aug :

            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                             std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),#
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))
            ])
            transform_train_mia = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
            ])

            transform_test_mia = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])

        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            
            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = torchvision.datasets.CIFAR100(
            data_root,
            train=True,
            download=True,
            transform=transform_train_mia
        )

        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = torchvision.datasets.CIFAR100(
            data_root,
            train=False,
            download=False,
            transform=transform_test_mia
        )

        train_set = torchvision.datasets.CIFAR100(
            data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            data_root,
            train=False,
            download=False,
            transform=transform_test
        )
        # 同样只保留前 50000 样本
        train_set = DatasetSplit(train_set, np.arange(0, 50000))
        idxs_a, idxs_b = stratified_split_dataset(train_set, frac=0.5, seed=42)
        # 用你的 DatasetSplit 包装子集
        subset_shadow = DatasetSplit(train_set, idxs_a)
        subset_target = DatasetSplit(train_set, idxs_b)

        print(f"Subset shadow: {len(subset_shadow)} 样本")
        print(f"Subset target: {len(subset_target)} 样本")
        if mode == True:#mode=self.trainShadowModel
            train_set = subset_shadow
            print("We are currently training a shadow model using a shadow dataset")
        else:
            train_set = subset_target
            print("We are currently training the target model using the target dataset")
    # -------------------- DermNet  --------------------
    if ds == 'dermnet':

        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)# 固定随机种子，保证每次切分一致
        #print(total_set[0].shape) # 19559, 3, 64, 64# 打印图像张量形状，如 (19559, 3, 64, 64)
        #print(total_set[1].shape) # 19559 # 打印标签长度，如 (19559,)
        # 随机打乱索引顺序
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]

        train_set=torch.utils.data.TensorDataset(
            total_set[0][0:15000],
            total_set[1][0:15000]
        )
        test_set=torch.utils.data.TensorDataset(
            total_set[0][-4000:],
            total_set[1][-4000:]
        )

        train_set_mia = train_set
        test_set_mia = test_set
        idxs_a, idxs_b = stratified_split_dataset(train_set, frac=0.5, seed=42)
        idxs_a, idxs_b = balance_by_tail_move(idxs_a, idxs_b)

        subset_shadow = DatasetSplit(train_set, idxs_a)
        subset_target = DatasetSplit(train_set, idxs_b)
        print(f"Subset shadow: {len(subset_shadow)} 样本")
        print(f"Subset target: {len(subset_target)} 样本")

        # # 验证各类样本数是否一致
        # from collections import Counter
        # labels_a = [subset_shadow[i][1] for i in range(len(subset_shadow))]
        # labels_b = [subset_b[i][1] for i in range(len(subset_b))]
        # print("影子数据集 每类样本数：", sorted(Counter(labels_a).items())[:5], "…")
        # print("目标数据集 每类样本数：", sorted(Counter(labels_b).items())[:5], "…")
        if mode == True:  # mode=self.trainShadowModel
            train_set = subset_shadow
            print("We are currently training a shadow model using a shadow dataset")
        else:
            train_set = subset_target
            print("当We are currently training the target model using the target dataset")

    if iid:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)
    else:
        dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label

class WMDataset(Dataset):
    def __init__(self, root, labelpath, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.labelpath = labelpath
        self.labels = np.loadtxt(self.labelpath)
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)

def prepare_wm(datapath='/trigger/pics/', num_back=1, shuffle=True):
    
    triggerroot = datapath
    labelpath = '/home/lbw/Dataset/trigger/labels-cifar.txt'

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ]

    transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset(triggerroot, labelpath, wm_transform)
    
    dict_users_back = wm_iid(dataset, num_back, 100)

    return dataset, dict_users_back

def prepare_wm_indistribution(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    triggerroot = datapath
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.ToTensor()
    ]

    #transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset_indistribution(triggerroot, wm_transform)
    
    num_all = num_trigger * num_back 

    dataset = DatasetSplit(dataset, np.arange(0, num_all))
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back

def prepare_wm_new(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    wm_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(datapath, wm_transform)
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back



class WMDataset_indistribution(Dataset):
    def __init__(self, root, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = 5
        if index in self.cache:
            img = self.cache[index]
        else:
        
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)
