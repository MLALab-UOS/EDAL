# -*- coding: utf-8 -*-
import random
import numpy as np
import os
from utils_method import *

def acquire_AcqOnly(args):
    args.model.eval()

    oodLst_idx = []

    # 1-1) Sampling
    print('...Acquisition Only')
    numberOfchoose_data = args.pool_data.size(0) if args.pool_data.size(0) <= args.pool_subset else args.pool_subset
    pool_subset_dropout = torch.from_numpy(
        np.asarray(random.sample(range(0, args.pool_data.size(0)), numberOfchoose_data))).long()
    pool_data_dropout = args.pool_data[pool_subset_dropout]
    pool_target_dropout = args.pool_target[pool_subset_dropout]

    pool_subset_idx = pool_subset_dropout.numpy()

    numOfSVHN = args.pool_subset - np.sum(args.isCifar10_pool[pool_subset_idx])
    print(f'20K중 {numOfSVHN}개의 OOD가 sampling')
    

    # 1-2) Acquisition
    points_of_interest = args.acquisition_function(pool_data_dropout, pool_target_dropout, args)
    points_of_interest = points_of_interest.detach().cpu().numpy()
    pool_index = np.flip(points_of_interest.argsort()[::-1][:int(args.numQ)], axis=0)

    pool_idx_wtoutOOD = pool_index.copy()


    # 1-3) svhn 데이터(OOD) 체크하기
    for i in range(len(pool_index)):
        if args.isCifar10_pool[pool_subset_idx[pool_index[i]]] == 0:
            oodLst_idx.append(i)
    print('SVHN 데이터', len(oodLst_idx), '개 체크')
    
    if args.isReject:     # train data set에서는 제외시키기
        pool_idx_wtoutOOD = np.delete(pool_idx_wtoutOOD, oodLst_idx)
        
        if args.method == 'AcqOnly':
            if args.data == 'Cifar10':
                saveDir = os.path.join('Results', 'erase_num', 'CS-Ent_erase_num.txt')
            else:
                saveDir = os.path.join('Results', 'erase_num', 'OS-Ent_erase_num.txt')
        else:
            saveDir = os.path.join('Results', 'erase_num', 'OS-Enr_erase_num.txt')
        f = open(saveDir, 'a')
        f.write(f'{len(oodLst_idx)} / {numOfSVHN}, ')
        f.close()
        args.queryOOD.append(len(oodLst_idx))
        args.dropoutOOD.append(numOfSVHN)
    else:     # train data set에 no.10 class로 추가시키기
        saveDir = os.path.join('Results', 'check_num', 'check_num.txt')
        f = open(saveDir, 'a')
        f.write(f'{len(oodLst_idx)}, ')
        f.close()

        saveDir = os.path.join('Results', 'check_num', 'check_ratio.txt')
        f = open(saveDir, 'a')
        f.write(f'{len(oodLst_idx)} / {numOfSVHN}, ')
        f.close()
        args.queryOOD.append(len(oodLst_idx))
        args.dropoutOOD.append(numOfSVHN)

    # Entropy 측정을 위한 세팅 추가
    pool_idx_wtOOD = np.array(pool_index[oodLst_idx])
    pool_idx_wtOOD = torch.from_numpy(pool_idx_wtOOD).long()
    OOD_from_poolData = pool_data_dropout[pool_idx_wtOOD]
    OOD_from_poolTarget = pool_target_dropout[pool_idx_wtOOD]

    pool_index = torch.from_numpy(pool_index)
    pool_idx_wtoutOOD = torch.from_numpy(pool_idx_wtoutOOD)
    pooled_data = pool_data_dropout[pool_idx_wtoutOOD]
    pooled_target = pool_target_dropout[pool_idx_wtoutOOD]

    ID_from_poolData = pooled_data.clone()
    ID_from_poolTarget = pooled_target.clone()

    batch_size = pooled_data.shape[0]
    target1 = pooled_target.unsqueeze(1)
    y_onehot1 = torch.FloatTensor(batch_size, args.nb_classes)
    y_onehot1.zero_()
    target1_oh = y_onehot1.scatter_(1, target1, 1)
    pooled_target_oh = target1_oh.float()

    # 1-3) Remove from pool_data
    pool_data, pool_target = remove_pooled_points(args.pool_data, args.pool_target, pool_subset_dropout,
                                                  pool_data_dropout, pool_target_dropout, pool_index, args)
    
    args.pool_data = pool_data
    args.pool_target = pool_target
    args.pool_all = np.append(args.pool_all, pool_index)

    print('isCifar10_pool 개수', len(args.isCifar10_pool))

    args.ID_from_poolData = ID_from_poolData
    args.ID_from_poolTarget = ID_from_poolTarget
    args.OOD_from_poolData = OOD_from_poolData
    args.OOD_from_poolTarget = OOD_from_poolTarget

    return pooled_data, pooled_target, pooled_target_oh
