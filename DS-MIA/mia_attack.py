import torch
import torch.nn.functional as F
import models as models
import Metrics as metr
import os
import numpy as np
import MetricSequence as MS
from torch.utils.data import DataLoader, Subset

from attackMethodsFramework import AttackingWithShadowTraining_RNN
from utils.datasets import DatasetSplit

def process_global(mode,path,attack_x, attack_y,metricFlag):
    num_metrics = metricFlag.count('&') + 1
    if num_metrics == 1:
        targetData = MS.createLossTrajectories_Seq(attack_x, attack_y, num_metrics)
    else:
        targetData = MS.createMetricSequences(attack_x, attack_y, num_metrics)
    save_dir = os.path.join('.', mode, path)  # './{mode}/{path}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'global_data.pt')
    torch.save(targetData, save_path)
    print('The result has been saved to the corresponding file')
    return targetData

def process_client(mode,attack_x, attack_y, metricFlag, client_data, client_evaluation):
    num_metrics = metricFlag.count('&') + 1
    if num_metrics == 1:
        targetData = MS.createLossTrajectories_Seq(attack_x, attack_y, num_metrics)
    else:
        targetData = MS.createMetricSequences(attack_x, attack_y, num_metrics)
    save_dir=os.path.join('.', mode, "attack_data",f"client{client_data}")
    save_path=os.path.join(save_dir, f"client{client_data}_on_client{client_evaluation}.pt")
    torch.save(targetData, save_path)
    print('The result has been saved to the corresponding file\n')

def createAttackDataWithMetrics_clients(
        mode='',
        dataset='cifar100',
        class_num=100,
        modelFolderPath=f'./shadow/model_save',
        classifierType='alexnet',
        batch_size=100,
        metricFlag='loss',
        client_data=0,
        client_evaluation=0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Starting creation client {client_data} attackData on client{client_evaluation}")

    #   torch.save({ 'train_set': train_set, 'test_set': test_set, … }, 'data_splits.pth')
    saved = torch.load(
        f"{mode}/data_splits.pth",
        weights_only=False
    )

    train_set_g = saved['train_set']
    train_idxs = saved['train_idxs']
    val_idxs = saved['val_idxs']


    train_set = Subset(train_set_g, train_idxs[client_data])
    if mode == 'shadow':
        test_set = DatasetSplit(saved['test_set'], np.arange(0, 5000))
    elif mode == 'target':
        test_set = DatasetSplit(saved['test_set'], np.arange(5000, 10000))
    print(f'client{client_data} train data lens {len(train_set)} test data lens {len(test_set)}')
    n_out = class_num
    if batch_size > len(train_set):
        batch_size = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_noShuffle = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f"Successfully loaded data for client {client_data}!")
    agg_dir = os.path.normpath(os.path.join(modelFolderPath,dataset, classifierType,'client', f'client{client_evaluation}'))
    snap_files = sorted(
        [f for f in os.listdir(agg_dir) if f.endswith('.pkl')],
        key=lambda fn: int(fn[:-4])
    )

    models_snap = []
    for fn in snap_files:
        path = os.path.join(agg_dir, fn)
        m = models.__dict__[classifierType](
            num_classes=class_num
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models_snap.append(m)
    print(f"Successfully loaded multiple model snapshots for client {client_evaluation}!")
    print(f"Start calculating the loss trajectory of data from client {client_data} on client {client_evaluation}")
    original_model = models_snap[0]
    distilledModeles = models_snap[1:]

    attack_x, attack_y = [], []
    classification_y = []
    losses = []
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        X_vector = X_vector.to(device)
        output = original_model(X_vector)
        out_y = output.detach().cpu()
        Y_vector_onehot = F.one_hot(Y_vector, n_out)
        loss_a_batch = metr.ComputeMetric(Y_vector, Y_vector_onehot, out_y, metricFlag='loss')
        loss_a_batch = loss_a_batch.squeeze()
        metrics_ori = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y, metricFlag=metricFlag)
        metrics_all = metrics_ori
        for m in distilledModeles:
            output_dis = m(X_vector)
            out_y_dis = output_dis.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector, n_out)
            metrics_dis = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y_dis, metricFlag=metricFlag)
            metrics_dis = metrics_dis
            metrics_all = torch.cat([metrics_all, metrics_dis], 1)

        num_metrics = metricFlag.count('&') + 1
        if num_metrics == 1:
            metrics_all = metrics_all

        elif num_metrics == 2:
            c, d = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d], 1)

        elif num_metrics == 3:
            c, d, e = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e], 1)

        elif num_metrics == 4:
            c, d, e, f = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e, f], 1)

        elif num_metrics == 5:
            c, d, e, f, g = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e, f, g], 1)

        attack_x.append(metrics_all)  # (batch_size, M)
        attack_y.append(np.ones(len(Y_vector)))
        classification_y.append(Y_vector)
        losses.append(loss_a_batch)

    if client_data == 0:
        print('Calculate the characteristics of test data')
        for step, (X_vector, Y_vector) in enumerate(test_loader):
            X_vector = X_vector.to(device)

            output = original_model(X_vector)
            out_y = output.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector, n_out)

            loss_a_batch = metr.ComputeMetric(Y_vector, Y_vector_onehot, out_y, metricFlag='loss')
            loss_a_batch = loss_a_batch.squeeze()

            metrics_ori = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y, metricFlag=metricFlag)
            metrics_all = metrics_ori
            for m in distilledModeles:
                output_dis = m(X_vector)
                out_y_dis = output_dis.detach().cpu()
                Y_vector_onehot = F.one_hot(Y_vector, n_out)
                metrics_dis = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y_dis, metricFlag=metricFlag)
                metrics_all = torch.cat([metrics_all, metrics_dis], 1)

            num_metrics = metricFlag.count('&') + 1
            if num_metrics == 1:
                metrics_all = metrics_all

            elif num_metrics == 2:
                c, d = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d], 1)

            elif num_metrics == 3:
                c, d, e = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e], 1)

            elif num_metrics == 4:
                c, d, e, f = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e, f], 1)

            elif num_metrics == 5:
                c, d, e, f, g = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e, f, g], 1)

            attack_x.append(metrics_all)  # (batch_size, M)
            attack_y.append(np.zeros(len(Y_vector)))  # 测试集标签为 0
            classification_y.append(Y_vector)
            losses.append(loss_a_batch)
    else:
        print('Do not calculate the characteristics of test data')

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classification_y = np.concatenate(classification_y)
    losses = np.concatenate(losses)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')
    losses = losses.astype('float32')
    print('AttackDataset for evaluation has {} samples'.format(len(classification_y)))
    process_client(mode,attack_x, attack_y, metricFlag, client_data, client_evaluation)


def createClientAttackDataOnGobal(mode='',client_num=5,client_data=0,dataset='cifar100',class_num=100,modelFolderPath='./model_save',classifierType='alexnet',batch_size=100,metricFlag='loss&max&sd&entropy&mentropy'
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Starting  creation  client {client_data} attackData on {mode} global model ")

    saved = torch.load(
        f"{mode}/data_splits.pth",
        weights_only=False
    )

    train_set_g = saved['train_set']
    train_idxs = saved['train_idxs']
    #val_idxs = saved['val_idxs']

    train_set = Subset(train_set_g, train_idxs[client_data])
    if mode == 'shadow':
        test_set = DatasetSplit(saved['test_set'], np.arange(0, 5000))
    elif mode == 'target':
        test_set = DatasetSplit(saved['test_set'], np.arange(5000, 10000))
    print(f'client{client_data} train data lens {len(train_set)} test data lens {len(test_set)}')

    n_out = class_num
    if batch_size > len(train_set):
        batch_size = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_noShuffle = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    agg_dir = os.path.join(modelFolderPath,dataset, classifierType, 'global')
    snap_files = sorted(
        [f for f in os.listdir(agg_dir) if f.endswith('.pkl')],
        key=lambda fn: int(fn[:-4])
    )

    models_snap = []
    for fn in snap_files:
        path = os.path.join(agg_dir, fn)
        m = models.__dict__[classifierType](
            num_classes=class_num
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models_snap.append(m)

    original_model = models_snap[0]
    distilledModeles = models_snap[1:]
    attack_x, attack_y = [], []
    classification_y = []
    losses = []
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        X_vector = X_vector.to(device)
        output = original_model(X_vector)
        out_y = output.detach().cpu()
        Y_vector_onehot = F.one_hot(Y_vector, n_out)
        loss_a_batch = metr.ComputeMetric(Y_vector, Y_vector_onehot, out_y, metricFlag='loss')
        loss_a_batch = loss_a_batch.squeeze()
        metrics_ori = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y, metricFlag=metricFlag)
        metrics_all = metrics_ori
        for m in distilledModeles:
            output_dis = m(X_vector)
            out_y_dis = output_dis.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector, n_out)
            metrics_dis = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y_dis, metricFlag=metricFlag)
            metrics_dis = metrics_dis
            metrics_all = torch.cat([metrics_all, metrics_dis], 1)


        num_metrics = metricFlag.count('&') + 1
        if num_metrics == 1:
            metrics_all = metrics_all

        elif num_metrics == 2:
            c, d = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d], 1)

        elif num_metrics == 3:
            c, d, e = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e], 1)

        elif num_metrics == 4:
            c, d, e, f = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e, f], 1)

        elif num_metrics == 5:
            c, d, e, f, g = torch.split(metrics_all, len(Y_vector), dim=0)
            metrics_all = torch.cat([c, d, e, f, g], 1)
        attack_x.append(metrics_all)  # (batch_size, M)
        attack_y.append(np.full(len(Y_vector), client_data, dtype=np.int32))
        classification_y.append(Y_vector)
        losses.append(loss_a_batch)
    if client_data == 0:
        print('Calculate the characteristics of test data')
        for step, (X_vector, Y_vector) in enumerate(test_loader):
            X_vector = X_vector.to(device)

            output = original_model(X_vector)
            out_y = output.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector, n_out)

            loss_a_batch = metr.ComputeMetric(Y_vector, Y_vector_onehot, out_y, metricFlag='loss')
            loss_a_batch = loss_a_batch.squeeze()

            metrics_ori = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y, metricFlag=metricFlag)
            metrics_all = metrics_ori
            for m in distilledModeles:
                output_dis = m(X_vector)
                out_y_dis = output_dis.detach().cpu()
                Y_vector_onehot = F.one_hot(Y_vector, n_out)
                metrics_dis = metr.ComputeMultiMetric(Y_vector, Y_vector_onehot, out_y_dis, metricFlag=metricFlag)
                metrics_all = torch.cat([metrics_all, metrics_dis], 1)

            num_metrics = metricFlag.count('&') + 1
            if num_metrics == 1:
                metrics_all = metrics_all

            elif num_metrics == 2:
                c, d = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d], 1)

            elif num_metrics == 3:
                c, d, e = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e], 1)

            elif num_metrics == 4:
                c, d, e, f = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e, f], 1)

            elif num_metrics == 5:
                c, d, e, f, g = torch.split(metrics_all, len(Y_vector), dim=0)
                metrics_all = torch.cat([c, d, e, f, g], 1)

            attack_x.append(metrics_all)  # (batch_size, M)
            #attack_y.append(np.zeros(len(Y_vector)))  # 测试集标签为 0
            attack_y.append(np.full(len(Y_vector), client_num, dtype=np.int32))  # 新：所有测试样本都标成 client_num
            classification_y.append(Y_vector)
            losses.append(loss_a_batch)
    else:
        print('Do not calculate the characteristics of test data')

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classification_y = np.concatenate(classification_y)
    losses = np.concatenate(losses)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')
    losses = losses.astype('float32')
    print('AttackDataset on {} global model for evaluation has {} train samples and {} test samples\n'.format(mode,len(train_set),len(test_set)))

    savepath=f'attack_data/client{client_data}'
    targetData=process_global(mode,savepath,attack_x, attack_y,metricFlag)
    return targetData,losses

def link_global_and_local_metrics(client_id: int,num_clients: int,global_data_path: str,local_dir: str,out_path: str
):
    global_list = torch.load(global_data_path, map_location='cpu')
    N = len(global_list)

    local_lists = []
    for j in range(num_clients):
        file_name = f"client{client_id}_on_client{j}.pt"
        path = os.path.join(local_dir, file_name)
        data = torch.load(path, map_location='cpu')

        local_lists.append([ traj[:, 1:].clone() for traj in data ])

    fused = []
    for i in range(N):
        g = global_list[i]
        loss_cols = [ local_lists[j][i] for j in range(num_clients) ]
        fused.append(torch.cat([g, *loss_cols], dim=1))

    torch.save(fused, out_path)
    print(f"Saved fused features for {N} samples to '{out_path}'.")

def merge_all_clients(num_clients=5,fused_dir: str = './model_save',out_path: str = './model_save/all_clients_fused.pt'):


    all_data = []

    for i in range(num_clients):

        fn = os.path.join(fused_dir,f'client{i}', f'final_client{i}.pt')

        data_i = torch.load(fn, map_location='cpu')
        print(f"Loaded {len(data_i)} samples from {fn}")
        all_data.extend(data_i)

    print(f"Total samples after merging: {len(all_data)}")

    torch.save(all_data, out_path)
    print(f"Saved merged data to '{out_path}'")

    return all_data


def FinalAttackDataGenerate(mode='',client_num=1,classifierType='',dataset='',class_num=1):
    for client_id in range(0,client_num):
        createClientAttackDataOnGobal(
            mode=mode,
            client_num=client_num,
            client_data=client_id,
            dataset=dataset,
            class_num=class_num,
            modelFolderPath=f'./{mode}/model_save',
            classifierType=classifierType,
            batch_size=100,
            metricFlag='loss&max&sd&entropy&mentropy'
        )
    for i in range(0, client_num):
        for j in range(0, client_num):
            createAttackDataWithMetrics_clients(
                mode=mode,
                dataset=dataset,
                class_num=class_num,
                modelFolderPath=f'./{mode}/model_save',
                classifierType=classifierType,
                batch_size=100,
                metricFlag='loss',#loss
                client_data=i,
                client_evaluation=j
            )
    for id in range(0,client_num):
        link_global_and_local_metrics(
            client_id=id,
            num_clients=client_num,
            global_data_path=f'./{mode}/attack_data/client{id}/global_data.pt',
            local_dir=f'./{mode}/attack_data/client{id}',
            out_path=f'./{mode}/attack_data/client{id}/final_client{id}.pt'
        )
    merge_all_clients(client_num,f'./{mode}/attack_data',f'./{mode}/attack_data/all_finalData.pt')

def main():
    client_num = 5
    num_class=client_num+1
    model='ResNet18'#alexnet  ResNet18
    dataset='cifar100'
    class_num=100
    FinalAttackDataGenerate('shadow',client_num,model,dataset,class_num)
    FinalAttackDataGenerate('target',client_num,model,dataset,class_num)
    modelType = 'rnnAttention' #'transformer'
    shadowData=torch.load("./shadow/attack_data/all_finalData.pt")#./shadow/attack_data/all_finalData
    targetData=torch.load("./target/attack_data/all_finalData.pt")
    targetY, pre_member_label, hidden_outputs = AttackingWithShadowTraining_RNN(shadowData,
                                                                                        targetData,
                                                                                        epochs=150,
                                                                                        batch_size=100,
                                                                                        modelType=modelType,
                                                                                        num_class=num_class)
    save_dir = os.path.join('.', "result",model, dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{num_class}_{modelType}.npz')
    np.savez(save_path, targetY=targetY, pre_member_label=pre_member_label)
    print('saved to:', os.path.abspath(save_path))

if __name__ == '__main__':
    main()