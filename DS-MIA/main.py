from utils.args import parser_args
from utils.datasets import *
import copy
import random
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import time
import models as models
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import quant


class FederatedLearning(Experiment):

    def __init__(self, args):
        super().__init__(args)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        self.optim=args.optim
        self.dp = args.dp
        self.defense=args.defense
        self.sigma = args.sigma
        self.sigma_sgd = args.sigma_sgd
        self.grad_norm=args.grad_norm
        self.save_dir = args.save_dir
        self.trainTargetModel=args.trainTargetModel
        self.trainShadowModel = args.trainShadowModel
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_root = args.data_root
 
        print('==> Preparing data...')
        self.train_set,self.test_set, self.train_set_mia, self.test_set_mia, self.dict_users, self.train_idxs, self.val_idxs = get_data(
            mode=self.trainShadowModel,
            dataset=self.dataset,
            data_root = self.data_root,
            iid = self.iid,
            num_users = self.num_users,
            data_aug=self.args.data_augment,
            noniid_beta=self.args.beta
            )

        os.makedirs(self.save_dir, exist_ok=True)
        if self.trainTargetModel:
            m = 'target'
        elif self.trainShadowModel:
            m = 'shadow'
        save_path = os.path.join(m, 'data_splits.pth')

        torch.save({
            'train_set': self.train_set,
            'test_set': self.test_set,
            'train_set_mia': self.train_set_mia,
            'test_set_mia': self.test_set_mia,
            'dict_users': self.dict_users,
            'train_idxs': self.train_idxs,
            'val_idxs': self.val_idxs,
        }, save_path)

        print(f"[+] Saved data & splits to {save_path}")

        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'dermnet':
            self.num_classes = 23

        print('==> Preparing model...')
        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],
                     'best_test_acc': -np.inf,# 最佳测试准确率初始为 -∞
                     'best_model': [],# 保存最佳模型参数
                     'local_loss': [], # 每轮本地损失均值
                     }

        self.construct_model()

        self.w_t = copy.deepcopy(self.model.state_dict())
        self.trainer = TrainerPrivate(self.model,
                                      self.train_set,
                                      self.device,
                                      self.dp,
                                      self.sigma,
                                      self.num_classes,
                                      self.defense,
                                      args.klam,
                                      args.up_bound,
                                      args.mix_alpha)
        self.tester = TesterPrivate(self.model, self.device)
              
    def construct_model(self):

        model = models.__dict__[self.args.model_name](num_classes=self.num_classes)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        torch.backends.cudnn.benchmark = True#
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    def train(self):
        train_ldr = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2)
        if self.trainShadowModel:
            if self.dataset == 'cifar100' or self.dataset == 'cifar10':
                test_set = DatasetSplit(self.test_set, np.arange(0, 5000))
            elif self.dataset=='dermnet':
                test_set = DatasetSplit(self.test_set, np.arange(0, 1500))
        elif self.trainTargetModel:
            if self.dataset == 'cifar100' or self.dataset == 'cifar10':
                test_set = DatasetSplit(self.test_set, np.arange(5000, 10000))
            elif self.dataset == 'dermnet':
                test_set = DatasetSplit(self.test_set, np.arange(1500, 3000))
        test_ldr = DataLoader(
            test_set,
            batch_size=self.batch_size ,
            shuffle=False,
            num_workers=2)
        local_train_ldrs = []
        if args.iid:
            for i in range(self.num_users):
                if args.defense=='instahide':
                    self.batch_size=len(self.dict_users[i])
                    # print("batch_size:",self.batch_size) 5000
                local_train_ldr = DataLoader(
                    DatasetSplit(self.train_set, self.dict_users[i]),
                    batch_size = self.batch_size,
                    shuffle=True,
                    num_workers=2)
                # print("len:",len(local_train_ldr)) 1
                local_train_ldrs.append(local_train_ldr)

        else:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(
                    DatasetSplit(self.train_set, self.dict_users[i]),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2)
                local_train_ldrs.append(local_train_ldr)


        total_time=0
        # {optim}_{lr_up}_{batch_size}_{时间戳}.log
        file_name = "_".join([
            'a',
            args.model_name,
            args.dataset,
            str(args.num_users),
            str(args.optim),
            str(args.lr_up),
            str(args.batch_size),
            str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))
        ])

        b=os.path.join(os.getcwd(), self.save_dir)
        if not os.path.exists(b):
            os.makedirs(b)
        fn=b+'/'+file_name+'.log'
        fn=file_name+'.log'
        fn=os.path.join(b,fn)
        print("training log saved in:",fn)

        lr_0=self.lr
        # 开始全局通信轮训练
        for epoch in range(self.epochs):

            global_state_dict=copy.deepcopy(self.model.state_dict())

            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(
                    range(self.num_users),
                    self.m,
                    replace=False
                )

            local_ws, local_losses,= [], []

            start = time.time()

            for idx in tqdm(
                    idxs_users,
                    desc=f"Epoch:{epoch+1}/{self.epochs}, lr:{self.lr:.6f}"):
                self.model.load_state_dict(global_state_dict)

                local_w, local_loss= self.trainer._local_update_noback(
                    local_train_ldrs[idx],
                    self.local_ep,
                    self.lr,
                    self.optim,
                    args.sampling_proportion)

                if self.trainShadowModel:
                    snapshot_dir = os.path.join(
                        'shadow',
                        "model_save",
                        self.args.dataset,  # e.g. "cifar100"
                        self.args.model_name,  # e.g. "alexnet"
                        "client",
                        f"client{idx}"
                    )
                elif self.trainTargetModel:
                    snapshot_dir = os.path.join(
                        'target',
                        "model_save",
                        self.args.dataset,  # e.g. "cifar100"
                        self.args.model_name,  # e.g. "alexnet"
                        "client",
                        f"client{idx}"
                    )

                os.makedirs(snapshot_dir, exist_ok=True)
                file_path = os.path.join(snapshot_dir, f"{epoch}.pkl")
                torch.save(local_w, file_path)

                if args.defense != 'none':
                    model_grads = {}#
                    for name, local_param in self.model.named_parameters():
                        if args.defense == 'quant': # TODO: 量化# 量化防御
                            model_grads[name]= local_w[name] - global_state_dict[name]
                            assert args.d_scale >= 1.0
                            model_grads[name]= quant(model_grads[name],int(args.d_scale))
                        elif args.defense == 'sparse':
                            model_grads[name]= local_w[name] - global_state_dict[name]
                            # print('d_scale: ', args.d_scale)
                            if model_grads[name].numel() > 1000:
                                threshold = torch.topk( torch.abs(model_grads[name]).reshape(-1), int(model_grads[name].numel() * (1 - args.d_scale))).values[-1]
                                # print(threshold)
                                # print(torch.sum(torch.abs(model_grads[name])<threshold)/model_grads[name].numel() )
                                model_grads[name]= torch.where(torch.abs(model_grads[name])<threshold, torch.zeros_like(model_grads[name]), model_grads[name])
                                # print("layer {} sparsity: {:.4f}".format(name, torch.sum(model_grads[name] == 0.0).float() / model_grads[name].numel()))
                        # elif args.defense == 'dp': # dp
                        #     model_grads[name]= local_w[name] - global_state_dict[name]
                        #     model_grads[name].add_(torch.randn_like(model_grads[name]), alpha=args.d_scale*torch.norm(model_grads[name], p=2))
                        #     #  + args.d_scale * torch.randn_like(local_w[name])
                        # elif args.defense == 'none': # 什么都不做
                        #     model_grads[name]= local_w[name] - global_state_dict[name]
                        for key,value in model_grads.items():
                            if key in local_w:
                                local_w[key] = global_state_dict[key] + model_grads[key]
                # test_loss, test_acc=self.trainer.test(test_ldr)
                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)
            if self.optim=="sgd":
                if self.args.lr_up=='common':
                    self.lr = self.lr * 0.99
                elif self.args.lr_up =='milestone':
                    if epoch in self.args.schedule_milestone:
                        self.lr *= 0.1
                else:
                    self.lr=lr_0 * (1 + math.cos(math.pi * epoch/ self.args.epochs)) / 2 
            else:
                pass
            client_weights = []
            for i in range(self.num_users):
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[i]))/len(self.train_set)
                client_weights.append(client_weight)

            self._fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)
            end = time.time()
            interval_time = end - start
            total_time+=interval_time

            if self.trainShadowModel:
                agg_dir = os.path.join(
                    'shadow',
                    "model_save",
                    self.args.dataset,
                    self.args.model_name,
                    "global"
                )
            elif self.trainTargetModel:
                agg_dir = os.path.join(
                    'target',
                    "model_save",
                    self.args.dataset,
                    self.args.model_name,
                    "global"
                )
            os.makedirs(agg_dir, exist_ok=True)
            torch.save(
                self.w_t,  # 或者 torch.save(self.model.state_dict(), ...)
                os.path.join(agg_dir, f"{epoch}.pkl")
            )

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(test_ldr)

                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))

                # use validation set as test set
                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f}'.format(
                    epoch+1,
                    self.epochs,
                    interval_time
                )
                )
                print(
                    "Train Loss {:.4f} --- Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean))
                print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(
                     acc_train_mean,
                     acc_val_mean,
                     self.logs['best_test_acc'])
                    )

                s = 'epoch:{}, lr:{:.5f}, val_acc:{:.4f}, ' \
                    'val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},' \
                    'time:{:.4f}, total_time:{:.4f}'.format(
                    epoch,
                    self.lr,
                    acc_val_mean,
                    loss_val_mean,
                    acc_train_mean,
                    loss_train_mean,
                    interval_time,
                    total_time
                )


        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}  '.format(
            self.logs['best_test_loss'],
            self.logs['best_test_acc']
             ))

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean

    def _fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])

        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i]

            self.w_t[k] = w_avg[k]

def main(args):

    if args.trainTargetModel:

        logs = {'net_info': None,
                'arguments': {
                    'frac': args.frac,
                    'local_ep': args.local_ep,
                    'local_bs': args.batch_size,
                    'lr_outer': args.lr_outer,
                    'lr_inner': args.lr,
                    'iid': args.iid,
                    'wd': args.wd,
                    'optim': args.optim,
                    'model_name': args.model_name,
                    'dataset': args.dataset,
                    'log_interval': args.log_interval,
                    'num_classes': args.num_classes,
                    'epochs': args.epochs,
                    'num_users': args.num_users }
                }
        save_dir = args.save_dir

        fl_target = FederatedLearning(args)

        if args.trainTargetModel:
            m='target'
        elif args.trainShadowModel:
            m='shadow'

        logg, time, best_test_acc, test_acc = fl_target.train()

        logs['net_info'] = logg
        logs['test_acc'] = test_acc
        logs['bp_local'] = True if args.bp_interval == 0 else False

        if not os.path.exists(save_dir + args.model_name + '/' + args.dataset):
            os.makedirs(save_dir + args.model_name + '/' + args.dataset)


    elif args.trainShadowModel:

        logs = {'net_info': None,
                'arguments': {
                    'frac': args.frac,
                    'local_ep': args.local_ep,
                    'local_bs': args.batch_size,
                    'lr_outer': args.lr_outer,
                    'lr_inner': args.lr,
                    'iid': args.iid,
                    'wd': args.wd,
                    'optim': args.optim,
                    'model_name': args.model_name,
                    'dataset': args.dataset,
                    'log_interval': args.log_interval,
                    'num_classes': args.num_classes,
                    'epochs': args.epochs,
                    'num_users': args.num_users
                }
                }
        save_dir = args.save_dir

        fl_shadow = FederatedLearning(args)

        logg, time, best_test_acc, test_acc = fl_shadow.train()

        logs['net_info'] = logg
        logs['test_acc'] = test_acc
        logs['bp_local'] = True if args.bp_interval == 0 else False

        if not os.path.exists(save_dir + args.model_name + '/' + args.dataset):
            os.makedirs(save_dir + args.model_name + '/' + args.dataset)



    return
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
if __name__ == '__main__':
    args = parser_args()
    print(args)
    setup_seed(args.seed)
    main(args)