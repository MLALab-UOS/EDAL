# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import math

from utils_data import *
from utils_al import *

parser = argparse.ArgumentParser(description='Active Learning with Data Augmentation')

parser.add_argument('--isSimpleTest', default=False, type=bool, help='lenet if True / ResNet if False')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs before acquiring points')
parser.add_argument('--epochs2', default=100, type=int, help='number of epochs after acquiring points')
parser.add_argument('--Gepochs', default=10, type=int, help='number of epochs for training Generator')
parser.add_argument('--model', default=None, help='classifier model')
parser.add_argument('--optimizer', default=None, help='optimizer of classifier')
parser.add_argument('--criterion', default=None, help='criterion of classifier')
parser.add_argument('--model_scheduler', default=None, help='model scheduler')
parser.add_argument('--best_acc', default=0., type=float, help='best accuracy')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--pool_batch_size', default=100, type=int, help='batch_size for calculating score in pool data')
parser.add_argument('--test_batch_size', default=100, type=int, help='test_batch_size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for learning rate')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for learning rate')
parser.add_argument('--ai', default=10, type=int, help='acquisition_iterations')
parser.add_argument('--di', default=1, type=int, help='dropout_iterations')
parser.add_argument('--numQ', default=1500, type=int, help='number of queries')

parser.add_argument('--isInit', default='False', type=str, help='True=Initialize / False=NoInitialize')
parser.add_argument('--rs', default=123, type=int, help='random seed / 123, 456, 789, 135, 246')
parser.add_argument('--rangeLayerMixFrom', default=1, type=int, help='range of layers to apply manifold mixup')
parser.add_argument('--rangeLayerMixTo', default=2, type=int, help='range of layers to apply manifold mixup')
parser.add_argument('--data', default='Cifar10', type=str, help='data name / Fashion, SVHN, Cifar10')
parser.add_argument('--acqType', default='MAX_ENTROPY', type=str, help='acquisition type')
parser.add_argument('--method', default='AcqOnly', type=str,\
                    help='methodology / VOS')
parser.add_argument('--alpha', default=2.0, type=float, help='alpha for Mixup')
parser.add_argument('--numLambda', default=5, type=int, help='the number of sampling lambda')
parser.add_argument('--didsharp', default='False', type=str, help='sharpening if true / no sharpening if false')
parser.add_argument('--sharp', default=1.0, type=float, help='sharpening parameter')

# Vk를 구하는데 사용되는 parameter: VOS 코드 참조
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=400)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=1000)
parser.add_argument('--loss_weight', type=float, default=0.1)
parser.add_argument('--isEstimateVk', type=bool, default=True)
# 쿼리셋 옵션 추가
parser.add_argument('--query_opt', type=str, default='origin', help='origin, top, middle, bottom, random')
parser.add_argument('--crit_opt', type=str, default='freqVk', help='freqVk, Intersec')

# OOD data를 train set에 포함시키지 않는 옵션
parser.add_argument('--isReject', type=bool, default=True)
parser.add_argument('--isOODloader', type=bool, default=False)
parser.add_argument('--oodRatio', default=-1, type=float, help='0.2, 0.4, 0.6')

args = parser.parse_args()

torch.set_printoptions(precision=10)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

rs = args.rs
seed_torch(rs)

cuda = True

if args.data == 'Fashion':
    args.input_dim, args.input_height, args.input_width = 1, 28, 28
    args.nb_classes = 10
if args.data == 'SVHN' or args.data == 'Cifar10':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 10
if args.data == 'Cifar100':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 100
    args.numQ = 100
    args.batch_size = 100
    args.isInit = 'True'
if args.data == 'Cifar10-SVHN':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 10
    args.sample_number = 1000
    args.start_epoch = 40
if args.data == 'Cifar10-Split':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 2
    args.sample_number = 1000
    args.start_epoch = 40

if args.method == 'VOS' and (args.data == 'Cifar10-SVHN' or args.data == 'Cifar10-Split'):
    # Vk를 구하는데 사용되는 변수들 선언: VOS 코드 참조
    args.weight_energy = torch.nn.Linear(args.nb_classes, 1).cuda()
    torch.nn.init.uniform_(args.weight_energy.weight)
    args.data_dict = torch.zeros(args.nb_classes, args.sample_number, 512).cuda()
    args.number_dict = {}
    for i in range(args.nb_classes):
        args.number_dict[i] = 0
    args.logistic_regression = torch.nn.Linear(1, 2)
    args.logistic_regression = args.logistic_regression.cuda()
    # optimizer의 parameter를 추가하는 부분은 utils_data.py에서 지정한다.


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

##################################
# Loading Data
# #################################

if args.data == 'Cifar10-SVHN':
    numberOfid = 46000
    oodClass = 10
elif args.data == 'Cifar10-Split':
    numberOfid = 9200
    oodClass = 8

if args.oodRatio != -1:
    args.limitOOD = math.ceil(numberOfid*args.oodRatio / (1-args.oodRatio) / oodClass)
else:
    args.limitOOD = None
if args.oodRatio == 0.6 and args.data == 'Cifar10-SVHN':
    args.limitOOD = None
    args.trigger = True
estStr = 'Estmate' if args.isEstimateVk else 'notEstimate'

train_loader_all_Cifar10, train_loader_all_SVHN, test_loader_Cifar10, test_loader_SVHN, trainset_all_Cifar10, trainset_all_SVHN, testset_Cifar10, testset_SVHN = make_loader(args, kwargs)
train_data, train_target, pool_data, pool_target, test_data, test_target = prepare_data_Prime(train_loader_all_Cifar10, train_loader_all_SVHN, test_loader_Cifar10, test_loader_SVHN, trainset_all_Cifar10, trainset_all_SVHN, testset_Cifar10, testset_SVHN, args)

args.train_data = train_data
args.pool_data = pool_data
args.pool_target = pool_target

##################################
# Run file
# #################################
def main():
    start_time = time.time()

    if args.data == 'Cifar10-SVHN':
        numberOfid = 46000
        oodClass = 10
    elif args.data == 'Cifar10-Split':
        numberOfid = 9200
        oodClass = 8

    if args.oodRatio != -1:
        args.limitOOD = math.ceil(numberOfid*args.oodRatio / (1-args.oodRatio) / oodClass)
    else:
        args.limitOOD = None
    estStr = 'Estmate' if args.isEstimateVk else 'notEstimate'

    args.saveDir = os.path.join('Results', args.data, str(args.rs), args.method, str(args.alpha), str(args.numLambda), str(args.oodRatio), f'{str(args.query_opt)}_{str(args.crit_opt)}_{estStr}')
    args.modelDir = os.path.join('Models', args.data, str(args.rs), args.method, str(args.alpha), str(args.numLambda), str(args.oodRatio), f'{str(args.query_opt)}_{str(args.crit_opt)}_{estStr}')

    if not os.path.exists(args.modelDir):
        print('Model Directory Constructed Successfully!!!')
        os.makedirs(args.modelDir)

    train_loader, test_loader = initialize_train_set(train_data, train_target, test_data, test_target, args)

    args.train_loader, args.train_data, args.train_target = train_loader, train_data, train_target
    args.test_loader, args.test_data, args.test_target = test_loader, test_data, test_target

    batch_size = args.train_data.shape[0]
    target = train_target.unsqueeze(1)
    y_onehot = torch.FloatTensor(batch_size, args.nb_classes)
    y_onehot.zero_()
    target_oh = y_onehot.scatter_(1, target, 1)
    args.train_target_oh = target_oh.float()

    args.layer_mix = None
    init_model(args, first=True)

    print_number_of_data(args)

    # ood Sample 저장하기
    args.VkSample = []
    # query에 포함되는 OOD 개수 및 dropout 된 개수 저장
    args.queryOOD = []
    args.dropoutOOD = []

    print("<Training without acquisition>")
    for epoch in range(1, args.epochs + 1):
        cls_loss, num_batch, num_data = train(epoch, args)
        print('Classifier Loss:', cls_loss / num_data)
        if epoch > args.start_epoch and args.method == 'VOS':
            enr_loss = train_enr(epoch, args)
            print('energyExtractor Loss:', enr_loss)
        if epoch % 10 == 0:
            test_loss, test_acc, best_acc = test(epoch, args)
        if epoch == args.epochs:
            print('Test Loss = %f\n' % (test_loss))
            print('Test Accuracy = %.2f%%\n' % (test_acc))
            print('Best Accuracy = %.2f%%\n' % (best_acc))
            args.test_loss = test_loss
            args.test_acc = test_acc


    print("<Acquiring points>")
    test_acc_hist = acquire_points(args.acqType, train_data, train_target, pool_data, pool_target, args)    # AL iteration part 시작.

    print('=========================')
    end_time = time.time()
    print_total_time(start_time, end_time)
    print('\n')

if __name__ == '__main__':
    main()
