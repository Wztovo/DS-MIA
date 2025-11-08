from numpy import random
import numpy as np

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid_MIA(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    all_idx0=all_idxs
    train_idxs=[]
    val_idxs=[]
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )

        train_idxs.append(list(dict_users[i] ))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0)-dict_users[i]))
    return dict_users, train_idxs, val_idxs

from typing import Dict, List, Tuple, Union

def inspect_partition(dataset, train_idxs, num_classes):

    y = np.array([dataset[i][1] for i in range(len(dataset))])
    K = len(train_idxs)
    M = np.zeros((K, num_classes), dtype=int)  # clients x classes

    for k, idxs in enumerate(train_idxs):
        cls = y[idxs]
        M[k] = np.bincount(cls, minlength=num_classes)

    totals = M.sum(1)
    P = (M + 1e-12) / totals[:, None]
    g = M.sum(0); g = g / g.sum()

    ent = (-np.sum(P * np.log(P), axis=1) / np.log(num_classes))
    l1  = np.abs(P - g).sum(1) / 2
    missing = (M == 0).sum(1)

    # print("每端总样本数:", totals.tolist())
    # print("每端归一化熵(0~1，越小越偏):", np.round(ent, 3).tolist())
    # print("每端与全局分布的L1距离(0~1，越大越偏):", np.round(l1, 3).tolist())
    # print("每端缺失的类别数:", missing.tolist())
    return M

def cifar_beta(
    dataset,
    beta: float,
    num_users: int,
    seed = None,
    min_size_per_client: int = 1,
    max_retries: int = 50,
    as_set: bool = False,
) -> Tuple[Dict[int, Union[list, set]], List[List[int]], List[List[int]]]:
    rng = np.random.default_rng(seed)
    y_list = []
    for i in range(len(dataset)):
        _, yi = dataset[i]
        y_list.append(int(yi))
    y = np.asarray(y_list, dtype=np.int32)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in classes}

    for attempt in range(max_retries):
        client_indices: List[List[int]] = [[] for _ in range(num_users)]

        for c in classes:
            idxs = idx_by_class[c]
            if not idxs:
                continue
            rng.shuffle(idxs)

            props = rng.dirichlet([beta] * num_users)
            counts = rng.multinomial(len(idxs), props)

            start = 0
            for k, cnt in enumerate(counts):
                if cnt > 0:
                    client_indices[k].extend(idxs[start:start + cnt])
                    start += cnt

        if all(len(lst) >= min_size_per_client for lst in client_indices):
            break
    else:
        needs = [(i, min_size_per_client - len(client_indices[i]))
                 for i in range(num_users) if len(client_indices[i]) < min_size_per_client]
        needs.sort(key=lambda x: x[1], reverse=True)
        for i, deficit in needs:
            if deficit <= 0:
                continue
            donors = sorted([(j, len(client_indices[j])) for j in range(num_users) if j != i],
                            key=lambda x: x[1], reverse=True)
            for j, size_j in donors:
                can_give = max(0, size_j - min_size_per_client)
                if can_give <= 0:
                    continue
                take = min(deficit, can_give)
                move = rng.choice(client_indices[j], size=take, replace=False)
                move_set = set(int(t) for t in move)
                client_indices[j] = [t for t in client_indices[j] if t not in move_set]
                client_indices[i].extend(list(move_set))
                deficit -= take
                if deficit <= 0:
                    break

    train_idxs: List[List[int]] = [sorted(lst) for lst in client_indices]
    dict_users = {i: set(train_idxs[i]) for i in range(num_users)}

    all_idxs_set = set(range(len(dataset)))
    val_idxs = [sorted(list(all_idxs_set - dict_users[i])) for i in range(num_users)]
    inspect_partition(dataset, train_idxs, num_classes=100)
    return dict_users, train_idxs, val_idxs


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

