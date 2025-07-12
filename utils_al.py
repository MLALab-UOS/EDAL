import time
import os
from scipy.stats import mode
import copy
import torch.utils.data as data_utils

from utils_data import *
from method_AcqOnly import *
from method_VOS import *


def getAcquisitionFunction(name):
    if name == "MAX_ENTROPY":
        return max_entroy_acquisition
    elif name == "energy_basedScore":
        return energy_score_acquisition
    else:
        print("ACQUSITION FUNCTION NOT IMPLEMENTED")
        raise ValueError

def acquire_points(argument, train_data, train_target, pool_data, pool_target, args, random_sample=False):

    args.pool_all = np.zeros(shape=(1))

    args.acquisition_function = getAcquisitionFunction(args.acqType)    # Energy Score를 acquisition score로 사용하기 위해 설정하는 부분

    args.test_acc_hist = [args.test_acc]
    args.train_loss_hist = [args.test_loss]
    args.mean_arr = []
    args.var_arr = []

    print(f'ood Ratio디버그: {1 - np.sum(args.isCifar10_pool) / len(args.isCifar10_pool)}')

    for i in range(args.ai):
        st = time.time()
        args.pool_subset = 20000    # 전체 unlabeled set 중에서 20,000개의 데이터만 랜덤으로 가져와 score를 측정할 예정
        print('---------------------------------')
        print("Acquisition Iteration " + str(i+1))
        print_number_of_data(args)

        print('(step1) Choose useful data')

        if args.method == 'AcqOnly':
            pooled_data, pooled_target, pooled_target_oh = acquire_AcqOnly(args)

            args.VkSample = []  # 매 AL round에 대해서 Vk sample 초기화

            if 1:   # Entropy 측정하기
                ID_from_poolData = args.ID_from_poolData
                ID_from_poolTarget = args.ID_from_poolTarget
                OOD_from_poolData = args.OOD_from_poolData
                OOD_from_poolTarget = args.OOD_from_poolTarget
                
                print(f'디버그: query 중 ID 개수: {ID_from_poolData.size(0)}, OOD 개수: {OOD_from_poolData.size(0)}, 합: {ID_from_poolData.size(0) + OOD_from_poolData.size(0)}')
                print(f'디버그: query 중 ID 개수: {len(ID_from_poolTarget)}, OOD 개수: {len(OOD_from_poolTarget)}, 합: {len(ID_from_poolTarget) + len(OOD_from_poolTarget)}')

                if len(ID_from_poolData) != 0:
                    entropy_of_ID = max_entroy_acquisition(ID_from_poolData, ID_from_poolTarget, args)
                    entropy_of_ID = entropy_of_ID.detach().cpu().numpy()
                    
                    print(f'디버그 - ID entropy 개수: {len(entropy_of_ID)}')
                    print(f'About ID: {min(entropy_of_ID):.4f} ~ {max(entropy_of_ID):.4f}, 평균: {np.mean(entropy_of_ID):.4f}, 분산: {np.var(entropy_of_ID)}')
                    args.mean_arr.append(np.mean(entropy_of_ID))
                    args.var_arr.append(np.var(entropy_of_ID))
                
                if len(OOD_from_poolData) != 0:
                    entropy_of_OOD = max_entroy_acquisition(OOD_from_poolData, OOD_from_poolTarget, args)
                    entropy_of_OOD = entropy_of_OOD.detach().cpu().numpy()

                    print(f'디버그 - OOD entropy 개수: {len(entropy_of_OOD)}')
                    print(f'About OOD: {min(entropy_of_OOD):.4f} ~ {max(entropy_of_OOD):.4f}, 평균: {np.mean(entropy_of_OOD):.4f}, 분산: {np.var(entropy_of_OOD)}')

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100' or args.data == 'Cifar10-SVHN':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader

            # if i == 9:
            #     # 디버그 코드 등장이요
            #     imgDict = {}
            #     c10_label = ['0_airplane', '1_automobile', '2_bird', '3_cat', '4_deer',
            #                 '5_dog', '6_frog', '7_horse', '8_ship', '9_truck']
            #     print('이미지 개수', len(args.train_loader.dataset))
            #     for j in range(len(args.train_loader.dataset)):
            #         img = args.train_loader.dataset[j][0] # Tensor형태의 이미지. [C, H, W]
            #         targetK = int(args.train_loader.dataset[j][1].numpy())

            #         # 폴더 만들기
            #         if not os.path.exists(f'./loader_img/{c10_label[targetK]}'):
            #             os.makedirs(f'./loader_img/{c10_label[targetK]}')
            #         if not targetK in imgDict.keys():
            #             imgDict[targetK] = 1
            #         else:
            #             imgDict[targetK] += 1

            #         img = img.numpy() # tensor -> numpy
            #         img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]

            #         plt.xticks([], [])
            #         plt.yticks([], [])
            #         plt.imshow(img)
            #         k = j + 1
            #         plt.savefig(f'./loader_img/{c10_label[targetK]}/{k:04}_{imgDict[targetK]:04}.png')
            #         plt.clf()
            #     print(imgDict)
            #     # 디버그 코드 퇴장이요

            total_train_loss = 0.0
            if args.isInit == 'True':
                init_model(args, first=False)
            for epoch in range(args.epochs2):
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                total_train_loss += train_loss_labeled/num_data_labeled
            total_train_loss /= args.epochs2

        if args.method == 'VOS':    # Energy Score를 활용한 Active Learning
            pooled_data, pooled_target, pooled_target_oh = acquire_VOS(args, i)

            args.VkSample = []  # 매 AL round에 대해서 Vk sample 초기화

            if 1:   # Entropy 측정하기
                ID_from_poolData = args.ID_from_poolData
                ID_from_poolTarget = args.ID_from_poolTarget
                OOD_from_poolData = args.OOD_from_poolData
                OOD_from_poolTarget = args.OOD_from_poolTarget
                
                print(f'디버그: query 중 ID 개수: {ID_from_poolData.size(0)}, OOD 개수: {OOD_from_poolData.size(0)}, 합: {ID_from_poolData.size(0) + OOD_from_poolData.size(0)}')
                print(f'디버그: query 중 ID 개수: {len(ID_from_poolTarget)}, OOD 개수: {len(OOD_from_poolTarget)}, 합: {len(ID_from_poolTarget) + len(OOD_from_poolTarget)}')

                if len(ID_from_poolData) != 0:
                    entropy_of_ID = max_entroy_acquisition(ID_from_poolData, ID_from_poolTarget, args)
                    entropy_of_ID = entropy_of_ID.detach().cpu().numpy()
                    
                    print(f'디버그 - ID entropy 개수: {len(entropy_of_ID)}')
                    print(f'About ID: {min(entropy_of_ID):.4f} ~ {max(entropy_of_ID):.4f}, 평균: {np.mean(entropy_of_ID):.4f}, 분산: {np.var(entropy_of_ID)}')
                    args.mean_arr.append(np.mean(entropy_of_ID))
                    args.var_arr.append(np.var(entropy_of_ID))
                
                if len(OOD_from_poolData) != 0:
                    entropy_of_OOD = max_entroy_acquisition(OOD_from_poolData, OOD_from_poolTarget, args)
                    entropy_of_OOD = entropy_of_OOD.detach().cpu().numpy()

                    print(f'디버그 - OOD entropy 개수: {len(entropy_of_OOD)}')
                    print(f'About OOD: {min(entropy_of_OOD):.4f} ~ {max(entropy_of_OOD):.4f}, 평균: {np.mean(entropy_of_OOD):.4f}, 분산: {np.var(entropy_of_OOD)}')

                # print(f'디버그 ID ent: {entropy_of_ID}')
                # print(f'디버그 OOD ent: {entropy_of_OOD}')

                # np.save('./id', entropy_of_ID)
                # np.save('./ood', entropy_of_OOD)

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100' or args.data == 'Cifar10-SVHN':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader
            # if i == 0:
            #     # 디버그 코드 등장이요
            #     imgDict = {}
            #     c10_label = ['0_airplane', '1_automobile', '2_bird', '3_cat', '4_deer',
            #                 '5_dog', '6_frog', '7_horse', '8_ship', '9_truck']
            #     print('이미지 개수', len(args.train_loader.dataset))
            #     for j in range(len(args.train_loader.dataset)):    # for i in range(100):
            #         img = args.train_loader.dataset[j][0] # Tensor형태의 이미지. [C, H, W]
            #         targetK = int(args.train_loader.dataset[j][1].numpy())

            #         # 폴더 만들기
            #         if not os.path.exists(f'./loader_img/{c10_label[targetK]}'):
            #             os.makedirs(f'./loader_img/{c10_label[targetK]}')
            #         if not targetK in imgDict.keys():
            #             imgDict[targetK] = 1
            #         else:
            #             imgDict[targetK] += 1

            #         img = img.numpy() # tensor -> numpy
            #         img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]

            #         plt.xticks([], [])
            #         plt.yticks([], [])
            #         plt.imshow(img)
            #         k = j + 1
            #         plt.savefig(f'./loader_img/{c10_label[targetK]}/{k:04}_{imgDict[targetK]:04}.png')
            #         plt.clf()
            #     print(imgDict)
            #     # 디버그 코드 퇴장이요
            total_train_loss = 0.0
            total_trainEnr_loss = 0.0
            if args.isInit == 'True':
                init_model(args, first=False)
            for epoch in range(args.epochs2):
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)    # classifier train
                total_train_loss += train_loss_labeled/num_data_labeled
                if epoch > args.start_epoch:
                    if args.train_OODloader is not None and args.isOODloader:
                        print('디버그: ood로더 사용')
                        enr_loss = train_enr_Prime(epoch, args)       # energy extractor train
                    elif args.isEstimateVk:
                        print('디버그: vk 사용')
                        enr_loss = train_enr(epoch, args)
                    else:
                        continue
                    print('energyExtractor Loss:', enr_loss)
                    total_trainEnr_loss += enr_loss
            total_train_loss /= args.epochs2
            if (args.train_OODloader is not None and args.isOODloader) or args.isEstimateVk:
                total_trainEnr_loss /= args.epochs2
            print('energyExtractor 전체 Loss:', total_trainEnr_loss)


        print('(step3) Results')
        args.layer_mix = None
        test_loss, test_acc, best_acc = test(i, args)
        args.test_acc_hist.append(test_acc)
        args.train_loss_hist.append(total_train_loss)
        et = time.time()

        print('...Train loss = %.5f' % total_train_loss)
        print('...Test accuracy = %.2f%%' % test_acc)
        print('...Best accuracy = %.2f%%' % args.best_acc)
        print_total_time(st, et)

    if not os.path.exists(args.saveDir):
        print('Result Directory Constructed Successfully!!!')
        os.makedirs(args.saveDir)

    if args.method == 'AcqOnly':
        if args.data == 'Cifar10':
            np.save(args.saveDir + '/Closed-set_Entropy_test_acc.npy', np.asarray(args.test_acc_hist))
            np.save(args.saveDir + '/Closed-set_Entropy_train_loss.npy', np.asarray(args.train_loss_hist))
        else:
            np.save(args.saveDir + '/Open-set_Entropy_test_acc.npy', np.asarray(args.test_acc_hist))
            np.save(args.saveDir + '/Open-set_Entropy_train_loss.npy', np.asarray(args.train_loss_hist))
            np.save(args.saveDir + '/Open-set_Entropy_entMean.npy', np.asarray(args.mean_arr))
            np.save(args.saveDir + '/Open-set_Entropy_entVar.npy', np.asarray(args.var_arr))
            np.save(args.saveDir + '/Open-set_Entropy_queryOOD.npy', np.asarray(args.queryOOD)/args.numQ)
            np.save(args.saveDir + '/Open-set_Entropy_dropoutOOD.npy', np.asarray(args.dropoutOOD))
    else:   # VOS method를 사용했을 때, 결과 저장하기
        np.save(args.saveDir + f'/Open-set_Energy_{args.crit_opt}_test_acc.npy', np.asarray(args.test_acc_hist))
        np.save(args.saveDir + f'/Open-set_Energy_{args.crit_opt}_train_loss.npy', np.asarray(args.train_loss_hist))
        np.save(args.saveDir + f'/Open-set_Energy_{args.crit_opt}_entMean.npy', np.asarray(args.mean_arr))
        np.save(args.saveDir + f'/Open-set_Energy_{args.crit_opt}_entVar.npy', np.asarray(args.var_arr))
        np.save(args.saveDir + '/Open-set_Energy_queryOOD.npy', np.asarray(args.queryOOD)/args.numQ)
        np.save(args.saveDir + '/Open-set_Energy_dropoutOOD.npy', np.asarray(args.dropoutOOD))

    return args.test_acc_hist


def max_entroy_acquisition(pool_data_dropout, pool_target_dropout, args):
    idx = -1
    score_All = torch.tensor(torch.zeros(pool_data_dropout.size(0), args.nb_classes))

    data_size = pool_data_dropout.shape[0]
    num_batch = int(data_size / args.pool_batch_size)
    for idx in range(num_batch):
        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]
        target = pool_target_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]

        pool = data_utils.TensorDataset(batch, target)
        pool_loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)
        predictions = evaluate2('loader', pool_loader, None, args)
        predictions = predictions.cpu().detach()
        score_All[idx*args.pool_batch_size:(idx+1)*args.pool_batch_size, :] = predictions
    
    if pool_data_dropout.size(0) % args.pool_batch_size != 0:
        batch = pool_data_dropout[(idx+1)*args.pool_batch_size:]
        target = pool_target_dropout[(idx+1)*args.pool_batch_size:]

        pool = data_utils.TensorDataset(batch, target)
        pool_loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)
        predictions = evaluate2('loader', pool_loader, None, args)
        predictions = predictions.cpu().detach()
        score_All[(idx+1)*args.pool_batch_size:, :] = predictions

    Avg_Pi = torch.div(score_All, args.di)
    # clamp 함수를 사용한 nan 제거
    Avg_Pi = torch.clamp(Avg_Pi, min= 1e-45)
    Log_Avg_Pi = torch.log2(Avg_Pi)
    Entropy_Avg_Pi = - torch.mul(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = torch.sum(Entropy_Avg_Pi, 1)

    U_X = Entropy_Average_Pi

    points_of_interest = U_X.flatten()

    return points_of_interest


def energy_score_acquisition(pool_data_dropout, pool_target_dropout, args):
    idx = -1
    score_All = torch.tensor(torch.zeros(pool_data_dropout.size(0), 1))

    data_size = pool_data_dropout.shape[0]
    num_batch = int(data_size / args.pool_batch_size)
    for idx in range(num_batch):
        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]
        target = pool_target_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]

        pool = data_utils.TensorDataset(batch, target)
        pool_loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)
        predictions = evaluate_energy('loader', pool_loader, None, args)
        predictions = predictions.cpu().detach()
        score_All[idx*args.pool_batch_size:(idx+1)*args.pool_batch_size, :] = predictions.unsqueeze(1)
    
    if pool_data_dropout.size(0) % args.pool_batch_size != 0:
        batch = pool_data_dropout[(idx+1)*args.pool_batch_size:]
        target = pool_target_dropout[(idx+1)*args.pool_batch_size:]

        pool = data_utils.TensorDataset(batch, target)
        pool_loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)
        predictions = evaluate_energy('loader', pool_loader, None, args)
        predictions = predictions.cpu().detach()
        score_All[(idx+1)*args.pool_batch_size:, :] = predictions.unsqueeze(1)
        
    points_of_interest = score_All.flatten()

    return points_of_interest


def log_sum_exp(args, value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(args.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)