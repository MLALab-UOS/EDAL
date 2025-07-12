# -*- coding: utf-8 -*-
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.distributions import uniform
from model_ResNet import ResNet18

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
import torch.utils.data as data_utils
import torchvision
from torchvision import datasets, transforms

class SVHN_Unknown(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super(SVHN_Unknown, self).__init__(*args, **kwargs)
        # 모든 레이블을 10으로 설정 (unknown class)
        self.labels = np.ones_like(self.labels) * 10

def make_loader(args, kwargs):

    if args.data == 'Fashion':
        train_loader_all = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader_all, test_loader, None, None

    if args.data == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        trainset_all = torchvision.datasets.SVHN(
            root='data', split='train', download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        testset = torchvision.datasets.SVHN(
            root='data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader_all, test_loader, trainset_all, testset

    if args.data == 'Cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        train_loader_all_SVHN = None
        trainset_all_SVHN = None
        testset_SVHN = None
        test_loader_SVHN = None

        return train_loader_all,  train_loader_all_SVHN, test_loader, test_loader_SVHN, trainset_all, trainset_all_SVHN, testset, testset_SVHN

    if args.data == 'Cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all = torchvision.datasets.CIFAR100(
            root='data', train=True, download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR100(
            root='data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader_all, test_loader, trainset_all, testset

    if args.data == 'Cifar10-SVHN':
        # Cifar10
        transform_train_Cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all_Cifar10 = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train_Cifar10)
        train_loader_all_Cifar10 = torch.utils.data.DataLoader(
            trainset_all_Cifar10, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test_Cifar10 = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset_Cifar10 = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform_test_Cifar10)
        test_loader_Cifar10 = torch.utils.data.DataLoader(
            testset_Cifar10, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        # SVHN
        transform_train_SVHN = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        trainset_all_SVHN = torchvision.datasets.SVHN(
            root='data', split='train', download=True, transform=transform_train_SVHN)
        train_loader_all_SVHN = torch.utils.data.DataLoader(
            trainset_all_SVHN, batch_size=args.batch_size, shuffle=True, **kwargs)

        testset_SVHN = None
        test_loader_SVHN = None

        return train_loader_all_Cifar10, train_loader_all_SVHN, test_loader_Cifar10, test_loader_SVHN, trainset_all_Cifar10, trainset_all_SVHN, testset_Cifar10, testset_SVHN
    
    if args.data == 'Cifar10-Split':
        # Cifar10
        transform_train_Cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all_Cifar10 = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train_Cifar10)
        train_loader_all_Cifar10 = torch.utils.data.DataLoader(
            trainset_all_Cifar10, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test_Cifar10 = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset_Cifar10 = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform_test_Cifar10)
        test_loader_Cifar10 = torch.utils.data.DataLoader(
            testset_Cifar10, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        trainset_all_SVHN = None
        train_loader_all_SVHN = None
        testset_SVHN = None
        test_loader_SVHN = None

        return train_loader_all_Cifar10, train_loader_all_SVHN, test_loader_Cifar10, test_loader_SVHN, trainset_all_Cifar10, trainset_all_SVHN, testset_Cifar10, testset_SVHN
    
    else:
        raise ValueError

def prepare_data(train_loader_all, test_loader, trainset_all, testset, args):

    if args.data == 'Fashion':

        train_data_all = train_loader_all.dataset.train_data
        train_target_all = train_loader_all.dataset.train_labels
        shuffler_idx = torch.randperm(train_target_all.size(0))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = test_loader.dataset.test_data
        test_target = test_loader.dataset.test_labels

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all.unsqueeze_(1)
        test_data.unsqueeze_(1)

        train_data_all = train_data_all.float()
        test_data = test_data.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :, :]
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")  # 10 x 10 x 1 x 28 x 28 --> 100 x 1 x 28 x 28
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")  # 10 x 10 x 1 x 28 x 28 --> 100 x 1 x 28 x 28
        pool_target = np.concatenate(pool_target, axis=0)

        mean = torch.tensor([0.1307])
        std = torch.tensor([0.3081])
        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = (test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'SVHN':

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.labels).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.labels).long()
        test_data = test_data.float().numpy()

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :, :]
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'Cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.targets).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.targets).long()
        test_data = test_data.float().numpy()
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :,:]
            data_i = np.transpose(data_i, (0, 3, 1, 2))
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_data_i = np.transpose(pool_data_i, (0, 3, 1, 2))
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'Cifar100':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.targets).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.targets).long()
        test_data = test_data.float().numpy()
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:10], :, :,:]
            data_i = np.transpose(data_i, (0, 3, 1, 2))
            target_i = train_target_all.numpy()[idx[0][0:10]]
            pool_data_i = train_data_all.numpy()[idx[0][10:], :, :, :]
            pool_data_i = np.transpose(pool_data_i, (0, 3, 1, 2))
            pool_target_i = train_target_all.numpy()[idx[0][10:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)
            # print(len(data_i))
            # print(len(train_data))

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

def prepare_data_Prime(train_loader_all_Cifar10, train_loader_all_SVHN, test_loader_Cifar10, test_loader_SVHN, trainset_all_Cifar10, trainset_all_SVHN, testset_Cifar10, testset_SVHN, args):
    if args.data == 'Cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        train_data_all = torch.Tensor(trainset_all_Cifar10.data)
        train_target_all = torch.Tensor(trainset_all_Cifar10.targets).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset_Cifar10.data)
        test_target = torch.Tensor(testset_Cifar10.targets).long()
        test_data = test_data.float().numpy()
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:400], :, :,:]
            data_i = np.transpose(data_i, (0, 3, 1, 2))
            target_i = train_target_all.numpy()[idx[0][0:400]]
            pool_data_i = train_data_all.numpy()[idx[0][400:], :, :, :]
            pool_data_i = np.transpose(pool_data_i, (0, 3, 1, 2))
            pool_target_i = train_target_all.numpy()[idx[0][400:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        args.isCifar10_pool = np.ones(len(pool_target), dtype= np.int64)
        print('디버그', np.sum(args.isCifar10_pool))
        print('디버그', len(args.isCifar10_pool))

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final
    
    
    if args.data == 'Cifar10-SVHN':
        # setup
        train_data = []
        train_target = []
        pool_data = []
        pool_target = []
        isCifar10_pool = []

        mean_Cifar10 = torch.tensor([0.4914, 0.4822, 0.4465])
        std_Cifar10 = torch.tensor([0.2023, 0.1994, 0.2010])
        mean_SVHN = torch.tensor([0.5, 0.5, 0.5])
        std_SVHN = torch.tensor([0.5, 0.5, 0.5])

        # Cifar10
        train_data_all_Cifar10 = torch.Tensor(trainset_all_Cifar10.data)
        train_target_all_Cifar10 = torch.Tensor(trainset_all_Cifar10.targets).long()
        shuffler_idx_Cifar10 = torch.randperm(len(train_target_all_Cifar10))
        train_data_all_Cifar10 = train_data_all_Cifar10[shuffler_idx_Cifar10]
        train_data_all_Cifar10 = train_data_all_Cifar10.float().numpy()     # .numpy()추가
        train_data_all_Cifar10 = np.transpose(train_data_all_Cifar10, (0, 3, 1, 2))     # transpose 추가
        train_target_all_Cifar10 = train_target_all_Cifar10[shuffler_idx_Cifar10]

        test_data_Cifar10 = torch.Tensor(testset_Cifar10.data)
        test_target_Cifar10 = torch.Tensor(testset_Cifar10.targets).long()
        test_data_Cifar10 = test_data_Cifar10.float().numpy()
        test_data_Cifar10 = np.transpose(test_data_Cifar10, (0, 3, 1, 2))

        # SVHN
        train_data_all_SVHN = torch.Tensor(trainset_all_SVHN.data)
        train_target_all_SVHN = torch.Tensor(trainset_all_SVHN.labels).long()
        shuffler_idx_SVHN = torch.randperm(len(train_target_all_SVHN))
        
        if args.trigger:
            selectNum = 69000
            selectIdx = np.random.permutation(selectNum)
            shuffler_idx_SVHN = shuffler_idx_SVHN[selectIdx]

        train_data_all_SVHN = train_data_all_SVHN[shuffler_idx_SVHN]
        train_data_all_SVHN = train_data_all_SVHN.float().numpy()       # .numpy()추가
        train_target_all_SVHN = train_target_all_SVHN[shuffler_idx_SVHN]

        # 정규화 진행하기
        # Cifar10
        train_data_all_Cifar10 = torch.from_numpy(train_data_all_Cifar10 / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        # SVHN
        train_data_all_SVHN = torch.from_numpy(train_data_all_SVHN / 255).float().sub(mean_SVHN[None, :, None, None]).div(std_SVHN[None, :, None, None])

        for i in range(0, args.nb_classes):
            # Cifar10
            arr_Cifar10 = np.array(np.where(train_target_all_Cifar10.numpy() == i))
            idx_Cifar10 = np.random.permutation(arr_Cifar10)
            data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][0:400], :, :,:]
            # data_i_Cifar10 = np.transpose(data_i_Cifar10, (0, 3, 1, 2))
            target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][0:400]]
            pool_data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][400:], :, :, :]
            # pool_data_i_Cifar10 = np.transpose(pool_data_i_Cifar10, (0, 3, 1, 2))
            pool_target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][400:]]

            if args.limitOOD is not None:
                # SVHN
                arr_SVHN = np.array(np.where(train_target_all_SVHN.numpy() == i))
                idx_SVHN = np.random.permutation(arr_SVHN)
                pool_data_i_SVHN = train_data_all_SVHN.numpy()[idx_SVHN[0][0:args.limitOOD], :, :, :]
                pool_target_i_SVHN = train_target_all_SVHN.numpy()[idx_SVHN[0][0:args.limitOOD]]
            else:
                # SVHN
                arr_SVHN = np.array(np.where(train_target_all_SVHN.numpy() == i))
                idx_SVHN = np.random.permutation(arr_SVHN)
                pool_data_i_SVHN = train_data_all_SVHN.numpy()[idx_SVHN[0][0:], :, :, :]
                pool_target_i_SVHN = train_target_all_SVHN.numpy()[idx_SVHN[0][0:]]

            # make train and pool set
            train_data.append(data_i_Cifar10)
            train_target.append(target_i_Cifar10)

            pool_data.append(pool_data_i_Cifar10)
            pool_data.append(pool_data_i_SVHN)
            pool_target.append(pool_target_i_Cifar10)
            pool_target.append(pool_target_i_SVHN)
            for _ in range(len(pool_target_i_Cifar10)):
                isCifar10_pool.append(1)
            for _ in range(len(pool_target_i_SVHN)):
                isCifar10_pool.append(0)

        args.isCifar10_pool = np.array(isCifar10_pool)
        # train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_data_final = torch.from_numpy(train_data)
        train_target = np.concatenate(train_target, axis=0)
        # pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_data_final = torch.from_numpy(pool_data)
        pool_target = np.concatenate(pool_target, axis=0)

        # train_data_final = torch.from_numpy(train_data / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        # pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)

        test_data_final = torch.from_numpy(test_data_Cifar10 / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        test_target_final = test_target_Cifar10

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final
    

    if args.data == 'Cifar10-Split':
        # setup
        train_data = []
        train_target = []
        test_data = []
        test_target = []
        pool_data = []
        pool_target = []
        isCifar10_pool = []

        mean_Cifar10 = torch.tensor([0.4914, 0.4822, 0.4465])
        std_Cifar10 = torch.tensor([0.2023, 0.1994, 0.2010])

        # Cifar10
        train_data_all_Cifar10 = torch.Tensor(trainset_all_Cifar10.data)
        train_target_all_Cifar10 = torch.Tensor(trainset_all_Cifar10.targets).long()
        shuffler_idx_Cifar10 = torch.randperm(len(train_target_all_Cifar10))
        train_data_all_Cifar10 = train_data_all_Cifar10[shuffler_idx_Cifar10]
        train_data_all_Cifar10 = train_data_all_Cifar10.float().numpy()     # .numpy()추가
        train_data_all_Cifar10 = np.transpose(train_data_all_Cifar10, (0, 3, 1, 2))     # transpose 추가
        train_target_all_Cifar10 = train_target_all_Cifar10[shuffler_idx_Cifar10]

        test_data_Cifar10 = torch.Tensor(testset_Cifar10.data)
        test_target_Cifar10 = torch.Tensor(testset_Cifar10.targets).long()
        test_data_Cifar10 = test_data_Cifar10.float().numpy()
        test_data_Cifar10 = np.transpose(test_data_Cifar10, (0, 3, 1, 2))

        # 정규화 진행하기
        # Cifar10
        train_data_all_Cifar10 = torch.from_numpy(train_data_all_Cifar10 / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        test_data_Cifar10 = torch.from_numpy(test_data_Cifar10 / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])

        for i in range(0, 2):   # 0, 1 classes are In-distribution class
            # Cifar10
            arr_Cifar10 = np.array(np.where(train_target_all_Cifar10.numpy() == i))
            idx_Cifar10 = np.random.permutation(arr_Cifar10)
            data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][0:400], :, :,:]
            # data_i_Cifar10 = np.transpose(data_i_Cifar10, (0, 3, 1, 2))
            target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][0:400]]
            pool_data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][400:], :, :, :]
            # pool_data_i_Cifar10 = np.transpose(pool_data_i_Cifar10, (0, 3, 1, 2))
            pool_target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][400:]]

            TEST_arr_Cifar10 = np.array(np.where(test_target_Cifar10.numpy() == i))
            TEST_idx_Cifar10 = np.random.permutation(TEST_arr_Cifar10)
            TEST_data_i_Cifar10 = test_data_Cifar10.numpy()[TEST_idx_Cifar10[0][0:], :, :,:]
            # data_i_Cifar10 = np.transpose(data_i_Cifar10, (0, 3, 1, 2))
            TEST_target_i_Cifar10 = test_target_Cifar10.numpy()[TEST_idx_Cifar10[0][0:]]

            # if args.limitOOD is not None:
            #     # SVHN
            #     arr_SVHN = np.array(np.where(train_target_all_SVHN.numpy() == i))
            #     idx_SVHN = np.random.permutation(arr_SVHN)
            #     pool_data_i_SVHN = train_data_all_SVHN.numpy()[idx_SVHN[0][0:args.limitOOD], :, :, :]
            #     pool_target_i_SVHN = train_target_all_SVHN.numpy()[idx_SVHN[0][0:args.limitOOD]]
            # else:
            #     # SVHN
            #     arr_SVHN = np.array(np.where(train_target_all_SVHN.numpy() == i))
            #     idx_SVHN = np.random.permutation(arr_SVHN)
            #     pool_data_i_SVHN = train_data_all_SVHN.numpy()[idx_SVHN[0][0:], :, :, :]
            #     pool_target_i_SVHN = train_target_all_SVHN.numpy()[idx_SVHN[0][0:]]

            # make train and pool set
            train_data.append(data_i_Cifar10)
            train_target.append(target_i_Cifar10)

            test_data.append(TEST_data_i_Cifar10)
            test_target.append(TEST_target_i_Cifar10)

            pool_data.append(pool_data_i_Cifar10)
            pool_target.append(pool_target_i_Cifar10)
            for _ in range(len(pool_target_i_Cifar10)):
                isCifar10_pool.append(1)
        
        for i in range(2, 10):      # 2 to 9 classes are Out-of-Distribution class
            arr_Cifar10 = np.array(np.where(train_target_all_Cifar10.numpy() == i))
            idx_Cifar10 = np.random.permutation(arr_Cifar10)

            if args.limitOOD is not None:
                pool_data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][0:args.limitOOD], :, :, :]
                pool_target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][0:args.limitOOD]]
            else:
                pool_data_i_Cifar10 = train_data_all_Cifar10.numpy()[idx_Cifar10[0][0:], :, :, :]
                pool_target_i_Cifar10 = train_target_all_Cifar10.numpy()[idx_Cifar10[0][0:]]
            
            pool_data.append(pool_data_i_Cifar10)
            pool_target.append(pool_target_i_Cifar10)
            for _ in range(len(pool_target_i_Cifar10)):
                isCifar10_pool.append(0)

        args.isCifar10_pool = np.array(isCifar10_pool)
        # train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_data_final = torch.from_numpy(train_data)
        train_target = np.concatenate(train_target, axis=0)

        test_data = np.concatenate(test_data, axis=0).astype("float32")
        test_data_final = torch.from_numpy(test_data)
        test_target = np.concatenate(test_target, axis=0)

        # pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_data_final = torch.from_numpy(pool_data)
        pool_target = np.concatenate(pool_target, axis=0)

        # train_data_final = torch.from_numpy(train_data / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)

        # test_data_final = torch.from_numpy(test_data_Cifar10 / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        test_target_final = torch.from_numpy(test_target)

        # pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean_Cifar10[None, :, None, None]).div(std_Cifar10[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, dataname, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.dataname = dataname
        self.transform = transform

    def __getitem__(self, index):
        if self.dataname == 'Cifar10' or self.dataname == 'Cifar100' or self.dataname == 'Cifar10-SVHN' or self.dataname == 'Cifar10-Split':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            x = self.tensors[0][index] * std[:, None, None] + mean[:, None, None]
        elif self.dataname == 'Fashion' or self.dataname == 'SVHN':
            x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


def initialize_train_set(train_data, train_target, test_data, test_target, args):
    if args.data == 'Cifar10' or args.data == 'Cifar100' or args.data == 'Cifar10-SVHN' or args.data == 'Cifar10-Split':
        transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        transform_train = None
    train_dataset = CustomTensorDataset(tensors=(train_data, train_target), dataname=args.data, transform=transform_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test = data_utils.TensorDataset(test_data, test_target)
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False)
    return train_loader, test_loader

class lenet(nn.Module):
    def __init__(self, args):
        super(lenet, self).__init__()
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.input_dim = args.input_dim
        self.class_num = args.nb_classes

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class lenetMultiChannel(nn.Module):
    def __init__(self, args):
        super(lenetMultiChannel, self).__init__()
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.input_dim = args.input_dim
        self.class_num = args.nb_classes

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def init_model(args, first=False):
    if first:
        args.model = ResNet18(args).cuda()
        if args.isInit == 'True':
            print('...Save model')
            torch.save(args.model.state_dict(), args.modelDir + '/model_init.pt')
        if args.method == 'VOS':    # main.py의 99번 line에서 언급한 부분: VOS 코드 참조
            args.optimizer_enr = optim.Adam(list(args.weight_energy.parameters()) + list(args.logistic_regression.parameters()),
                                            lr=args.lr, betas=(args.beta1, args.beta2))
        args.optimizer_cls = optim.Adam(args.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        args.criterion = nn.CrossEntropyLoss()
        args.best_acc = 0.
    else:
        args.model = ResNet18(args).cuda()
        if args.isInit == 'True':
            print('...Load model')
            args.model.load_state_dict(torch.load(args.modelDir + '/model_init.pt'))
        args.optimizer_cls = optim.Adam(args.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        args.criterion = nn.CrossEntropyLoss()

def train(epoch, args):

    args.model.train()
    num_batch = 0
    num_data = 0
    total_train_loss = 0.0

    for batch_idx, (batch, target) in enumerate(args.train_loader):
        num_batch += 1
        batch_size = batch.shape[0]
        num_data += batch_size

        # batch, target = Variable(batch), Variable(target)
        batch, target = batch.cuda(), target.cuda()

        args.optimizer_cls.zero_grad()
        if args.method == 'AdaMixup':
            bef, output = args.model(batch)
        else:
            output = args.model(batch, args)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)
        loss.backward()
        args.optimizer_cls.step()
        total_train_loss += loss.item() * batch_size
        
        # batch, target = batch.cuda(), target.cuda()

        # x, pen_x = args.model.forward_virtual(batch, args)
            
        # output = pen_x.detach().clone()     # 때어내고,
        # output.requires_grad_()             # 다시 backward 연산 가능

        # criterion = nn.CrossEntropyLoss()

        # loss = criterion(x, target)
        # loss.backward()
        # args.optimizer.step()
        # total_train_loss += loss.item() * batch_size

    return total_train_loss, num_batch, num_data

def train_enr(epoch, args):

    args.model.train()
    num_batch = 0
    num_data = 0
    total_train_loss = 0.0
    eye_matrix = torch.eye(512, device='cuda')
    tempVk = []
    cnt = 0

    # if epoch in [41, 100]:
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(Cls)')
    #     for name, child in args.model.named_children():
    #         for param in child.parameters():
    #             print(name, param)
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(Enr_weight)')
    #     print(f'weight: {args.weight_energy.weight}\nbias: {args.weight_energy.bias}')
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(logistic)')
    #     print(f'weight: {args.logistic_regression.weight}\nbias: {args.logistic_regression.bias}')
    #     print('------------------------------')

    for batch_idx, (batch, target) in enumerate(args.train_loader):
        num_batch += 1
        batch_size = batch.shape[0]
        num_data += batch_size

        batch, target = batch.cuda(), target.cuda()

        x_prime, pen_x = args.model.forward_virtual(batch, args)
        
        x = x_prime.detach().clone()
        x.requires_grad_(True)
        output = pen_x.detach().clone()     # 때어내고,
        output.requires_grad_(True)         # 다시 backward 연산 가능

        sum_temp = 0
        for index in range(args.nb_classes):
            sum_temp += args.number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        lr_reg_loss.requires_grad_(True)

        if sum_temp == args.nb_classes * args.sample_number and epoch < args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                args.data_dict[dict_key] = torch.cat((args.data_dict[dict_key][1:],
                                                    output[index].detach().view(1, -1)), 0)
        elif sum_temp == args.nb_classes * args.sample_number and epoch > args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                args.data_dict[dict_key] = torch.cat((args.data_dict[dict_key][1:],
                                                    output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(args.nb_classes):
                if index == 0:
                    X = args.data_dict[index] - args.data_dict[index].mean(0)
                    mean_embed_id = args.data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, args.data_dict[index] - args.data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                            args.data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(args.nb_classes):
                if not torch.isnan(mean_embed_id[index][0]):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((args.sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    # breakpoint()
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, args.select)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                else:
                    print('에폭:', epoch, '클래스', index, '에서 nan 발생')
                    ood_samples=[]
            if len(ood_samples) != 0:
                from utils_al import log_sum_exp
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(args, x, 1)   #copy를 활용한 값만 가져오기 - classifier와의 연결을 제거하기.
                # predictions_ood = args.model.linear(ood_samples) 디태치로 수정하기
                predictions_ood_prime = args.model.linear(ood_samples)
                predictions_ood = predictions_ood_prime.detach().clone()
                # for visualize
                if epoch in [41, 50, 60, 70, 80, 90, 100]:
                    temp = torch.logsumexp(predictions_ood, 1).unsqueeze(1).detach().cpu()
                    temp = temp.numpy()
                    for i in range(len(temp)):
                        # args.VkSample.append(temp[i][0])
                        tempVk.append(temp[i][0])
                    cnt += 1


                predictions_ood.requires_grad_(True)


                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(args, predictions_ood, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                        torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = args.logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())      # 여기서 backward classifier 제외
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if args.number_dict[dict_key] < args.sample_number:
                    args.data_dict[dict_key][args.number_dict[dict_key]] = output[index].detach()
                    args.number_dict[dict_key] += 1

        args.optimizer_enr.zero_grad()
        # breakpoint()
        # # 잠시 디버그
        # lr_reg_loss = args.loss_weight * lr_reg_loss + lr_reg_loss
        # lr_reg_loss.backward()
        
        # 현우 제안
        totLoss = args.loss_weight * lr_reg_loss
        totLoss.backward()

        args.optimizer_enr.step()

        # exponential moving average
        total_train_loss += float(totLoss)

    # if epoch in [41, 100]:
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(Cls)')
    #     for name, child in args.model.named_children():
    #         for param in child.parameters():
    #             print(name, param)
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(Enr_weight)')
    #     print(f'weight: {args.weight_energy.weight}\nbias: {args.weight_energy.bias}')
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(logistic)')
    #     print(f'weight: {args.logistic_regression.weight}\nbias: {args.logistic_regression.bias}')
    #     print('------------------------------')

    if cnt != 0:
        args.VkSample.append(tempVk)

    return total_train_loss

def train_enr_Prime(epoch, args):

    args.model.train()
    num_batch = 0
    num_data = 0
    total_train_loss = 0.0

    ood_latent_vector = []
    id_latent_vector = []

    iter = -1

    torch.autograd.set_detect_anomaly(True)

    from utils_al import log_sum_exp

    # if epoch in [41, 100]:
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(Cls)')
    #     for name, child in args.model.named_children():
    #         for param in child.parameters():
    #             print(name, param)
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(Enr_weight)')
    #     print(f'weight: {args.weight_energy.weight}\nbias: {args.weight_energy.bias}')
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이전(logistic)')
    #     print(f'weight: {args.logistic_regression.weight}\nbias: {args.logistic_regression.bias}')
    #     print('------------------------------')

    
    
    # task 1 - OOD feature extract
    for batch_idx, (batch, target) in enumerate(args.train_OODloader):
        num_batch += 1
        batch_size = batch.shape[0]
        num_data += batch_size

        batch, target = batch.cuda(), target.cuda()

        x_prime, _ = args.model.forward_virtual(batch, args)
        
        x = x_prime.detach().clone()
        x.requires_grad_(True)

        ood_latent_vector.append(x)
    
    setOOD = num_batch
    num_batch = 0
    
    # task 2 - ID feature extract
    for batch_idx, (batch, target) in enumerate(args.train_IDloader):
        num_batch += 1
        batch_size = batch.shape[0]
        num_data += batch_size

        batch, target = batch.cuda(), target.cuda()

        x_prime, _ = args.model.forward_virtual(batch, args)

        x = x_prime.detach().clone()
        x.requires_grad_(True)

        id_latent_vector.append(x)

    # setID = num_batch

    # numIter = setID if setID <= setOOD else setOOD
    
    # # train Energy extractor
    # for iter in range(numIter-1):
    #     lr_reg_loss = torch.zeros(1).cuda()[0]
    #     lr_reg_loss.requires_grad_(True)
        
    #     featureID = id_latent_vector[iter]
    #     featureOOD = ood_latent_vector[iter]

    #     energy_score_for_fg = log_sum_exp(args, featureID, 1)
    #     energy_score_for_bg = log_sum_exp(args, featureOOD, 1)

    #     input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
    #     labels_for_lr = torch.cat((torch.ones(len(featureID)).cuda(),
    #                             torch.zeros(len(featureOOD)).cuda()), -1)

    #     criterion = torch.nn.CrossEntropyLoss()
    #     output1 = args.logistic_regression(input_for_lr.view(-1, 1))
    #     lr_reg_loss = criterion(output1, labels_for_lr.long())

    #     args.optimizer_enr.zero_grad()
    #     # breakpoint()
    #     lr_reg_loss = args.loss_weight * lr_reg_loss + lr_reg_loss
    #     lr_reg_loss.backward()

    #     args.optimizer_enr.step()

    #     # exponential moving average
    #     total_train_loss += float(lr_reg_loss)
    
    # # 나머지 자투리 학습시키기
    # if setID >= setOOD:
    #     idTemp = id_latent_vector[iter+1:]
    #     ood = ood_latent_vector[iter+1]

    #     for i in range(len(idTemp)):
    #         if i == 0:
    #             temp = idTemp[i]
    #         else:
    #             torch.cat((temp, idTemp[i]), 0)
        
    #     featureID = temp
    #     featureOOD = ood
    # else:
    #     id = id_latent_vector[iter+1]
    #     oodTemp = ood_latent_vector[iter+1:]

    #     for i in range(len(oodTemp)):
    #         if i == 0:
    #             temp = oodTemp[i]
    #         else:
    #             torch.cat((temp, oodTemp[i]), 0)
        
    #     featureID = id
    #     featureOOD = temp
    
    # lr_reg_loss = torch.zeros(1).cuda()[0]
    # lr_reg_loss.requires_grad_(True)

    # energy_score_for_fg = log_sum_exp(args, featureID, 1)
    # energy_score_for_bg = log_sum_exp(args, featureOOD, 1)

    # input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
    # labels_for_lr = torch.cat((torch.ones(len(featureID)).cuda(),
    #                         torch.zeros(len(featureOOD)).cuda()), -1)

    # criterion = torch.nn.CrossEntropyLoss()
    # output1 = args.logistic_regression(input_for_lr.view(-1, 1))
    # lr_reg_loss = criterion(output1, labels_for_lr.long())

    # args.optimizer_enr.zero_grad()
    # # breakpoint()
    # lr_reg_loss = args.loss_weight * lr_reg_loss + lr_reg_loss
    # lr_reg_loss.backward()

    # args.optimizer_enr.step()

    # # exponential moving average
    # total_train_loss += float(lr_reg_loss)

    # if epoch in [41, 100]:
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(Cls)')
    #     for name, child in args.model.named_children():
    #         for param in child.parameters():
    #             print(name, param)
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(Enr_weight)')
    #     print(f'weight: {args.weight_energy.weight}\nbias: {args.weight_energy.bias}')
    #     print('------------------------------')
    #     print(f'Epoch: {epoch}_[Parameter freeze]: backward 이후(logistic)')
    #     print(f'weight: {args.logistic_regression.weight}\nbias: {args.logistic_regression.bias}')
    #     print('------------------------------')


    for i in range(len(id_latent_vector)):
        if i == 0:
            temp = id_latent_vector[i]
        else:
            torch.cat((temp, id_latent_vector[i]), 0)

    featureID = temp
    
    for i in range(len(ood_latent_vector)):
        if i == 0:
            temp = ood_latent_vector[i]
        else:
            torch.cat((temp, ood_latent_vector[i]), 0)

    featureOOD = temp


    lr_reg_loss = torch.zeros(1).cuda()[0]
    lr_reg_loss.requires_grad_(True)

    energy_score_for_fg = log_sum_exp(args, featureID, 1)
    energy_score_for_bg = log_sum_exp(args, featureOOD, 1)

    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
    labels_for_lr = torch.cat((torch.ones(len(featureID)).cuda(),
                            torch.zeros(len(featureOOD)).cuda()), -1)

    criterion = torch.nn.CrossEntropyLoss()
    output1 = args.logistic_regression(input_for_lr.view(-1, 1))
    lr_reg_loss = criterion(output1, labels_for_lr.long())

    args.optimizer_enr.zero_grad()
    # breakpoint()
    lr_reg_loss = args.loss_weight * lr_reg_loss + lr_reg_loss
    lr_reg_loss.backward()

    args.optimizer_enr.step()

    # exponential moving average
    total_train_loss += float(lr_reg_loss)

    return total_train_loss

def evaluate2(data_type, loader, data, args):

    predictions = []

    if data_type == 'loader':
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if args.method == 'AdaMixup':
                _, output = args.model(data)
            else:
                output = args.model(data, args)
            softmaxed = F.softmax(output, dim=1)
            predictions.append(softmaxed)
        predictions = torch.cat(predictions, dim=0)

    if data_type == 'batch':
        args.model.eval()
        data = data.cuda()
        if args.method == 'AdaMixup':
            _, output = args.model(data)
        else:
            output = args.model(data, args)
        softmaxed = F.softmax(output, dim=1)
        predictions.append(softmaxed)
        predictions = torch.cat(predictions, dim=0)

    return predictions


def evaluate_energy(data_type, loader, data, args):

    predictions = []

    if data_type == 'loader':
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if args.method == 'AdaMixup':
                _, output = args.model(data)
            else:
                # 모델에 넣기
                output = args.model(data, args)
            # energy score 구하기 & 전처리
            energy_score = torch.logsumexp(output, 1)
            energy_score.reshape(energy_score.shape[0],1)
            for i in range(len(energy_score)):
                predictions.append(energy_score[i].reshape(1))
        predictions = torch.cat(predictions, dim=0)

    if data_type == 'batch':
        from utils_al import log_sum_exp
        args.model.eval()
        data = data.cuda()
        if args.method == 'AdaMixup':
            _, output = args.model(data)
        else:
            # 모델에 넣기
            output = args.model(data, args)
        
        # energy score 구하기
        energy_score = log_sum_exp(output, 1)
        # logistic regression의 함수값 구하기
        out1 = args.logistic_regression(energy_score)
        # 최종 acquisiton score 구하기
        output = 1 / (1 + torch.exp(-out1))
        # (OOD일 확률, ID일 확률) 이므로 뒤에 값을 택한 후, 큰 값을 기준으로 topk 함수 이용
        output = output[1]

        predictions.append(output)
        predictions = torch.cat(predictions, dim=0)

    return predictions


def evaluate(loader, args, stochastic=False, predict_classes=False):
    if stochastic:
        args.model.train()
    else:
        args.model.eval()

    predictions = []
    test_loss = 0
    correct = 0

    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if args.method == 'AdaMixup':
            _, output = args.model(data)
        else:
            output = args.model(data, args)
        softmaxed = F.softmax(output.cpu(), dim=1)

        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(), axis=-1))
        else:
            predictions.extend(softmaxed.data.numpy())

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.data.max(1)[1]
        pred = pred.eq(target.data).cpu().data.float()
        correct += pred.sum()

    return test_loss, correct, predictions

def test(epoch, args):

    test_loss, correct, _ = evaluate(args.test_loader, args, stochastic=False)

    test_loss /= len(args.test_loader)
    test_acc = 100. * correct / len(args.test_loader.dataset)

    if test_acc > args.best_acc:
        args.best_acc = test_acc


    return test_loss, test_acc, args.best_acc

