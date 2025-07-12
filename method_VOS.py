# -*- coding: utf-8 -*-
import random
import numpy as np
import os
from utils_method import *
from scipy.stats import gaussian_kde
from visualize import *

def acquire_VOS(args, numOfai):

    args.model.eval()

    oodLst_idx = []

    vis_ID = []
    vis_OOD = []
    vis_Vk = args.VkSample # 지움

    print('디버그_vis_Vk 개수(7이여야 함)', len(vis_Vk))
    for i in range(len(vis_Vk)):
        print(f'디버그_{i+1}: sample의 개수_{len(vis_Vk[i])}')
        vis_Vk[i] = np.array(vis_Vk[i])
    print('print의 개수가 일치해야할듯')

    
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

    for i in range(len(points_of_interest)):
        if args.isCifar10_pool[pool_subset_idx[i]] == 1:
            vis_ID.append(points_of_interest[i])
        else:
            vis_OOD.append(points_of_interest[i])

    save_folder = f'./energy_vis/{numOfai+1}/{args.query_opt}_{args.crit_opt}/{args.rs}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_folder1 = f'./energy_value/{args.rs}'
    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)


    x_ID, y_ID, x_OOD, y_OOD = vis_ID_OOD_Energy(vis_ID, vis_OOD, numOfai, save_folder)

    for i in range(len(vis_Vk)):
        number = (i+4)*10
        if number == 40:
            number = 41
        
        x_Vk, y_Vk = vis_ALL_Energy(x_ID, y_ID, x_OOD, y_OOD, vis_Vk[i], number, save_folder)

        if i == len(vis_Vk) - 1 and args.crit_opt == 'freqVk' and args.query_opt == 'middle':
            np.save(save_folder1 + f'/{numOfai+1}_xID.npy', x_ID)
            np.save(save_folder1 + f'/{numOfai+1}_yID.npy', y_ID)
            np.save(save_folder1 + f'/{numOfai+1}_xOOD.npy', x_OOD)
            np.save(save_folder1 + f'/{numOfai+1}_yOOD.npy', y_OOD)
            np.save(save_folder1 + f'/{numOfai+1}_xVk.npy', x_Vk)
            np.save(save_folder1 + f'/{numOfai+1}_yVk.npy', y_Vk)

    if args.query_opt == 'origin':
        pool_index = np.flip(points_of_interest.argsort()[::-1][:int(args.numQ)], axis=0)
    elif args.query_opt == 'random':
        pool_index = np.random.choice(len(points_of_interest), args.numQ, replace= False)
    else:
        # arr = np.array(y_Vk) # 잠시 수정
        if args.isEstimateVk and args.crit_opt == 'freqVk':
            arr = np.array(y_Vk)
            maxIdx = np.argmax(arr)
            args.critValue = x_Vk[maxIdx]
            print(f'안녕제건: {args.critValue}')
        elif not args.isEstimateVk and args.crit_opt == 'freqVk':   # by Frequency of Vk
            if numOfai == 0:
                arr = np.array(y_Vk)
                maxIdx = np.argmax(arr)
                args.critValue = x_Vk[maxIdx]
            else:
                pass
        pool_index = np.flip(points_of_interest.argsort(), axis=0)
        logic = 1

        for cnt, idx in enumerate(pool_index):
            if args.crit_opt == 'freqVk':   # by Frequency of Vk
                temp = int(np.sign(points_of_interest[idx] - args.critValue))
                if logic * temp <= 0:
                    if args.query_opt == 'top':
                        minimum = int((cnt+1)-args.numQ) if cnt+1 >= args.numQ else 0
                        maximum = cnt+1
                    if args.query_opt == 'middle':
                        minimum = int((cnt+1)-int(args.numQ/2)) if cnt+1 >= int(args.numQ/2) else 0
                        maximum = int((cnt+1)+int(args.numQ/2)) if args.pool_subset - (cnt+1) >= int(args.numQ/2) else -1
                    if args.query_opt == 'bottom':
                        minimum = cnt+1
                        maximum = int((cnt+1)+args.numQ) if args.pool_subset - (cnt+1) >= args.numQ else -1
                    
                    idxLst = pool_index[minimum:maximum]
                    print(f'디버그: criterion 찾기 완료 / 위치: {cnt+1}')
                    break
            elif args.crit_opt == 'Intersec':   # by Intersection of ID and OOD
                kde_ID = gaussian_kde(vis_ID)
                kde_OOD = gaussian_kde(vis_OOD)

                args.critValue = kde_OOD(points_of_interest[idx])
                temp = int(np.sign(kde_ID(points_of_interest[idx]) - args.critValue))
                if logic * temp <= 0:
                    if args.query_opt == 'top':
                        minimum = int((cnt+1)-args.numQ) if cnt+1 >= args.numQ else 0
                        maximum = cnt+1
                    if args.query_opt == 'middle':
                        minimum = int((cnt+1)-int(args.numQ/2)) if cnt+1 >= int(args.numQ/2) else 0
                        maximum = int((cnt+1)+int(args.numQ/2)) if args.pool_subset - (cnt+1) >= int(args.numQ/2) else -1
                    if args.query_opt == 'bottom':
                        minimum = cnt+1
                        maximum = int((cnt+1)+args.numQ) if args.pool_subset - (cnt+1) >= args.numQ else -1
                    
                    idxLst = pool_index[minimum:maximum]
                    print(f'디버그: criterion 찾기 완료 / 위치: {cnt+1}')
                    break
            logic = temp
        pool_index = idxLst.copy()

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
            saveDir = os.path.join('Results', 'erase_num', str(args.query_opt), str(args.crit_opt), 'OS-Enr_erase_num.txt')  # 수정
        f = open(saveDir, 'a')
        f.write(f'{len(oodLst_idx)} / {numOfSVHN}, ')
        f.close()
        args.queryOOD.append(len(oodLst_idx))
        args.dropoutOOD.append(numOfSVHN)
    else:     # train data set에 no.10 class로 추가시키기   -> 이 세팅은 이제 사라짐
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
    pool_idx_wtoutOOD = torch.from_numpy(pool_idx_wtoutOOD).long()
    pooled_data = pool_data_dropout[pool_idx_wtoutOOD]
    pooled_target = pool_target_dropout[pool_idx_wtoutOOD]

    ID_from_poolData = pooled_data.clone()
    ID_from_poolTarget = pooled_target.clone()

    # ood_loader를 위한 dataset 만들기
    pooled_OODdata = pool_data_dropout[pool_idx_wtOOD]
    pooled_OODtarget = pool_target_dropout[pool_idx_wtOOD]
    if args.data == 'Cifar10-SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    else:
        transform_train = None
    
    if len(pooled_OODdata) != 0 and args.isOODloader:
        train_OODdataset = data_utils.TensorDataset(pooled_OODdata, pooled_OODtarget)
        train_OODloader = data_utils.DataLoader(train_OODdataset, batch_size=args.batch_size, shuffle=True)
        args.train_OODloader = train_OODloader
    else:
        args.train_OODloader = None
    
    # id_loader를 위한 dataset 만들기
    if args.data == 'Cifar10' or args.data == 'Cifar100' or args.data == 'Cifar10-SVHN':
        transform_train = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    else:
        transform_train = None
    train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if len(ID_from_poolData) != 0:
        train_IDdataset = data_utils.TensorDataset(ID_from_poolData, ID_from_poolTarget)
        train_IDloader = data_utils.DataLoader(train_IDdataset, batch_size=args.batch_size, shuffle=True)
        args.train_IDloader = train_IDloader

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
    args.poolg_all = np.append(args.pool_all, pool_index)

    print('isCifar10_pool 개수', len(args.isCifar10_pool))

    args.ID_from_poolData = ID_from_poolData
    args.ID_from_poolTarget = ID_from_poolTarget
    args.OOD_from_poolData = OOD_from_poolData
    args.OOD_from_poolTarget = OOD_from_poolTarget

    return pooled_data, pooled_target, pooled_target_oh
