import os, sys, time#时间相关的包
import numpy as np#np
import matplotlib#画图用的
import subprocess
import argparse#用于加载args.yaml参数的
import torch
import torch.cuda as cutorch
import torchvision.transforms as transforms
import torchvision.models.resnet
import torchvision#CV\DL专用的胡数据库
from random import sample#参数的包，
# sample(a[list],n) 功能：从序列a中随机抽取n个元素，并将n个元素生以list形式返回
matplotlib.use('agg')
from ReIDdatasets import *
import yaml#用于加载args.yaml参数的
import math
from collections import OrderedDict#collections模块为我们提供了OrderdDict,用于获取一个有序的字典对象
#参数初始化和部分与训练测试无关的简单操作的定义

class BaseOptions(object):
    """
    base options for deep learning for Re-ID.
    parse basic arguments by parse(), print all the arguments by print_options()
    """
    def __init__(self):#初始化
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#formatter_class - 用于自定义帮助文档输出格式的类
        self.args = None

        self.parser.add_argument('--save_path', type=str, default='debug', help='Folder to save checkpoints and log.')
        self.parser.add_argument('--resume', default='runs/duke', type=str, help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--gpu', type=str, default='0', help='gpu used.')

    def parse(self):#加载参数的函数，返回的是参数表self.args
        self.args = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        with open(os.path.join(self.args.save_path, 'args.yaml')) as f:#打开参数的配置文件，文件路径拼接
            extra_args = yaml.load(f)#提取参数，从文件中提取参数，并放到extra_args中
        self.args = argparse.Namespace(**vars(self.args), **extra_args)#从yaml文件中导入参数，将**extra_args的参数放入**vars(self.args)中
        #*self._args 表示接受元组类参数；**kwargs 表示接受字典类参数；
        return self.args#返回参数列表

    def print_options(self, logger):
        #打印出参数列表
        logger.print_log("")#logger.print_log 输出到屏幕，将log日志写入文件
        logger.print_log("----- options -----".center(120, '-'))#center，居中
        args = vars(self.args)#如果对象具有__dict__属性，则vars（）函数返回给定对象的__dict__属性
        #如果vars()函数没有输入参数,用法与locals()相同,以字典形式返回当前位置的全部局部变量名
        string = ''
        for i, (k, v) in enumerate(sorted(args.items())):#k是参数名，v是参数值
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(args.items())-1:#每3个参数对输出一次
                logger.print_log(string)
                string = ''
        logger.print_log("".center(120, '-'))
        logger.print_log("")


class Logger(object):#输出到屏幕，将log日志写入文件
    def __init__(self, save_path):#__init__函数中，self后面的位置可添加其他参数
        if not os.path.isdir(save_path):
            os.makedirs(save_path)#判断文件路径是否存在，如果不存在，则创建
        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_log(self, string):#输入到文件的同时，输出到缓冲区
        self.file.write("{}\n".format(string))
        self.file.flush()#在文件关闭前刷新缓冲区
        print(string)


class AverageMeter(object):
    #这个是用于计算当前的运行数据的一些统计量，当前值，平均值，总和、方差等
    """Compute and store the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():#时间格式化
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def extract_features(loader, model, index_feature=None, return_numpy=True):#index_feature=0
    """
    应该是利用预训练或者训练过的模型来进行提取特征
    extract features for the given loader using the given model
    if loader.dataset.require_views is False, the returned 'views' are empty.
    :param loader: a ReIDDataset that has attribute require_views
    :param model: returns a tuple containing the feature or only return the feature. if latter, index_feature be None (tuple,feature)/(feature)
    model can also be a tuple of nn.Module, indicating that the feature extraction is multi-stage.
    in this case, index_feature should be a tuple of the same size.
    :param index_feature: in the tuple returned by model, the index of the feature.
    if the model only returns feature, this should be set to None.
    :param return_numpy: if True, return numpy array; otherwise return torch tensor
    :return: features, labels, views, np array
    """
    if type(model) is not tuple:
        models = (model,)
        indices_feature = (index_feature,)
    else:
        assert len(model) == len(index_feature)
        models = model
        indices_feature = index_feature
    for m in models:
        m.eval()#在model(test_datasets)之前，需要加上model.eval()
        #这是model中含有batch normalization层所带来的的性质

    labels = []
    views = []
    features = []#list

    require_views = loader.dataset.require_views
    for i, data in enumerate(loader):#也是批量提取特征的
        imgs = data[0].cuda()
        label_batch = data[1]
        inputs = imgs
        for m, feat_idx in zip(models, indices_feature):#index_feature=0
            with torch.no_grad():#强制之后的内容不进行梯度 计算图构建，既然是提取特征，当然不需要进行梯度计算
                output_tuple = m(inputs)
            feature_batch = output_tuple if feat_idx is None else output_tuple[feat_idx]
            inputs = feature_batch#看样子是会进行循环迭代的

        features.append(feature_batch)
        labels.append(label_batch)
        if require_views:
            view_batch = data[2]#使用状态信息
            views.append(view_batch)
    features = torch.cat(features, dim=0)#dim=0按行拼接， dim=1按列拼接，转Tensor
    labels = torch.cat(labels, dim=0)
    views = torch.cat(views, dim=0) if require_views else views#if成立时输出if前面的公式，否则输出else后面的东西
    if return_numpy:
        return np.array(features.cpu()), np.array(labels.cpu()), np.array(views.cpu())
    else:
        return features, labels, views


def create_stat_string(meters):#某些状态信息
    stat_string = ''
    for stat, meter in meters.items():
        stat_string += '{} {:.3f}   '.format(stat, meter.avg)
    return stat_string


def convert_secs2time(epoch_time):#秒转时间格式
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None,
                 probe_views=None, ignore_MAP=True):#计算CMC或mAP
    #dist = cdist(gallery_features, probe_features, metric='cosine')
    #dist 二维矩阵，行数为gallery_features_num,列数为probe_features_num,此项目采用的是余弦距离
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :param ignore_MAP: is True, only compute cmc
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    """
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)
    '''
    array和asarray都可将结构数据转换为ndarray类型。
    但是主要区别就是当数据源是ndarray时，
    array仍会copy出一个副本，占用新的内存，但asarray不会
    '''
    is_view_sensitive = False#不使用状态信息
    num_gallery = gallery_labels.shape[0]#获得gallery_set的数目
    num_probe = probe_labels.shape[0]#获得query_set的数目
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'#两个同时成立时，则继续往下执行，否则报错，错误信息为逻辑语句后面的信息
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):#计算CMC曲线
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]#取出对应的query_feature与gallery_feature的距离
        probe_label = probe_labels[i]#取出对应的query_feature的标签
        gallery_labels_ = gallery_labels#取出对应的gallery_feature的标签
        if is_view_sensitive:#考虑是否使用状态信息，如果状态信息和标签信息相同时，需要执行一些操作，这里不知道是不是缩小特征边界之类的
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view#判断状态是否相同
            is_correct = gallery_labels == probe_label#判断标签是否相同
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_) #返回的是元素值从小到大排序后的索引值的数组
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list) #返回的是数组中，非零元素的位置
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1#将正确的位置以及之后的位置设置为1
        cmc[:, i] = cmc_

        if not ignore_MAP:#计算AP
            num_correct = positions_correct.shape[0]#gallery_set中正确的个数
            for j in range(num_correct):#不知道这里是不是仅考虑两个命中的AP
                last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
                current_precision = float(j + 1) / float(positions_correct[j] + 1)
                ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC * 100, MAP * 100


def occupy_gpu_memory(gpu_ids, maximum_usage=None, buffer_memory=2000):
    """
    As pytorch is dynamic, you might wanna take enough GPU memory to avoid OOM when you run your code
    in a messy server.指定最大内存使用
    if maximum_usage is specified, this function will return a dummy buffer which takes memory of
    (current_available_memory - (maximum_usage - current_usage) - buffer_memory) MB.
    otherwise, maximum_usage would be replaced by maximum usage till now, which is returned by
    torch.cuda.max_memory_cached()
    :param gpu_ids:
    :param maximum_usage: float, measured in MB
    :param buffer_memory: float, measured in MB
    :return:
    """
    n_gpu = int((len(gpu_ids)-1)/2+1)
    for i in range(n_gpu):
        gpu_id = int(gpu_ids[i*2])
        if maximum_usage is None:
            maximum_usage = cutorch.max_memory_cached()
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader'])
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split(b'\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        available_memory = gpu_memory_map[gpu_id]
        if available_memory < buffer_memory+1000:
            print('Gpu memory has been mostly occupied (although maybe not by you)!')
        else:
            memory_to_occupy = int((available_memory - (maximum_usage - cutorch.memory_cached(i)/1024/1024) - buffer_memory))
            dim = int(memory_to_occupy * 1024 * 1024 * 8 / 32)
            x = torch.zeros(dim, dtype=torch.int)
            x.pin_memory()
            x_ = x.cuda(device=torch.device('cuda:{}'.format(i)))
            print('Occupied {}MB extra gpu memory in gpu{}.'.format(memory_to_occupy, gpu_id))
            del x_


def save_checkpoint(trainer, epoch, save_path, is_best=False):#保存某个训练时间中的最佳的模型
    logger = trainer.logger
    trainer.logger = None
    if not os.path.isdir(os.path.dirname(save_path)):	#dirname 返回文件路径，isdir判断路径是否是目录
        os.mkdir(os.path.dirname(save_path))#创建目录
    torch.save((trainer, epoch), save_path)
    if is_best:
        best_path = save_path + '.best'
        torch.save((trainer, epoch), best_path)
    trainer.logger = logger


def load_checkpoint(args, logger):
    #加载出某个保存的模型
    """
    load a checkpoint (containing a trainer and an epoch number) and assign a logger to the loaded trainer.
    the differences between the loaded trainer.args and input args would be print to logger.
    :param args:
    :param logger:
    :return:
    """
    load_path = args.resume#路径
    assert os.path.isfile(load_path)#判断是否为文件夹
    logger.print_log("=> loading checkpoint '{}'".format(load_path))#格式化输出路径
    (trainer, epoch) = torch.load(load_path)#加载模型
    trainer.logger = logger

    old_args = trainer.args
    trainer.args = args

    attributes = vars(args)#返回对象object的属性和属性值的字典对象
    old_attributes = vars(old_args)
    for name, value in attributes.items():
        if name == 'resume' or name == 'gpu':
            continue
        if name in old_attributes:#检查attributes与old_attributes是否有相同的属性
            old_value = old_attributes[name]
            if old_value != value:#属性值不等时
                logger.print_log("args.{} was {} but now is replaced by the newly specified one: {}.".format(name, old_value,
                                                                                                             value))
        else:
            logger.print_log("args.{} was not specified in the checkpoint.".format(name))
    return trainer, epoch#加载模型，检查模型参数与最佳参数的区别，并打印出相关参数


def compute_accuracy(predictions, labels):
    #计算每一个batch的精度
    """
    compute classification accuracy, measured by percentage.
    :param predictions: tensor. size = N*d
    :param labels: tensor. size = N
    :return: python number, the computed accuracy
    """
    predicted_labels = torch.argmax(predictions, dim=1)#返回最大的元素的索引值
    n_correct = torch.sum(predicted_labels == labels).item()
    batch_size = torch.numel(labels)
    acc = float(n_correct) / float(batch_size)
    return acc * 100


def eval_acc(dist, gallery_labels, probe_labels):
    #计算rank-1的精度，这里是用于CFP的
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)

    ranking_table = np.argsort(dist, axis=0)# axis=0对a按列进行排序 ，axis=1对a按行进行排序
    r1_idx = ranking_table[0]
    infered_labels = gallery_labels[r1_idx]
    acc = (infered_labels == probe_labels).mean()*100
    return acc


def partition_params(module, strategy, *desired_modules):
    #暂时看不懂
    """
    将参数分为两部分，bn和其他；如果是具体的话，就要看desired_modules的名字
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: strings, each corresponds to a specific module
    :return: two lists
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            #判断该模块是不是归一化模块
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))#将bn层的参数放到 desired_params集合里面
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))#将desired_modules层的参数放到 desired_params集合里面
    else:
        assert False, 'unknown strategy: {}'.format(strategy)#未知策略，啥不干，就不放参数
    all_params_set = set(module.parameters())#将所有层的参数放到all_params_set
    other_params_set = all_params_set.difference(desired_params_set)#将desired_params_set以外的层的参数放到这个
    desired_params = list(desired_params_set)#转list
    other_params = list(other_params_set)
    return desired_params, other_params


def get_reid_dataloaders(dataset, img_size, crop_size, padding, batch_size):
    #reid数据加载器
    """
    get train/gallery/probe dataloaders.
    :return:
    """

    train_data = Market('{}.mat'.format(dataset), state='train')
    gallery_data = Market('{}.mat'.format(dataset), state='gallery')
    probe_data = Market('{}.mat'.format(dataset), state='probe')

    mean = train_data.return_mean() / 255.0
    std = train_data.return_std() / 255.0

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),#水平翻转，大小裁剪，随机裁剪，转为Tensor类型，归一化
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)

    return train_loader, gallery_loader, probe_loader


def get_mpie_dataloaders(dataset_path, img_size, crop_size, padding, batch_size):
    #mpie数据加载器，暂时不看
    """
    get train/gallery/probe dataloaders.
    :return:
    """

    train_data = MultiPie(dataset_path, state='train')
    gallery_data = MultiPie(dataset_path, state='gallery')
    probe_data = MultiPie(dataset_path, state='probe')

    mean = np.array([91.4953, 103.8827, 131.0912])
    std = np.array([1.0, 1.0, 1.0])
    # mean = train_data.return_mean() / 255
    # std = train_data.return_std() / 255
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    """Assuming using CAFFE pretrained model, thus transform input to BGR,
    and do not divide std."""

    back_to_256 = transforms.Lambda(lambda tensor: tensor*255)
    to_bgr = transforms.Lambda(lambda tensor: tensor[[2, 1, 0], :, :])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(),
         back_to_256, to_bgr, transforms.Normalize(mean, std)])
         # transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.CenterCrop(crop_size), transforms.ToTensor(),
         back_to_256, to_bgr, transforms.Normalize(mean, std)])
         # transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)

    return train_loader, gallery_loader, probe_loader


def get_cfp_dataloaders(dataset_path, img_size, crop_size, padding, batch_size, test_set):
    # cfp数据加载器，暂时不看
    """
    get train/gallery/probe dataloaders.
    :return:
    """

    train_data = CFP(dataset_path, [x for x in range(10) if x != test_set])
    test_data = CFP(dataset_path, [test_set])
    protocol = test_data.return_protocol()

    mean = np.array([91.4953, 103.8827, 131.0912])
    std = np.array([1.0, 1.0, 1.0])

    """Assuming using CAFFE pretrained model, thus transform input to BGR,
    and do not divide std."""

    back_to_256 = transforms.Lambda(lambda tensor: tensor*255)
    to_bgr = transforms.Lambda(lambda tensor: tensor[[2, 1, 0], :, :])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(),
         back_to_256, to_bgr, transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.CenterCrop(crop_size), transforms.ToTensor(),
         back_to_256, to_bgr, transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    test_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=2, pin_memory=True)

    return train_loader, test_loader, protocol


def eval_CFP(sim_same, sim_diff):
    #CFP相关指标评估，暂时不看
    """
    :param sim_same: 1d tensor, every entry is a similarity
    :param sim_diff: 1d tensor, every entry is a similarity
    :return: acc, EER, AUC
    """

    FAR = []
    FRR = []
    TAR = []
    FP_acc = []
    for threshold in torch.arange(0.0, 1.0, 0.01):
        true_p = (sim_same >= threshold).sum().type(torch.Tensor)
        false_n = (sim_same < threshold).sum().type(torch.Tensor)
        true_n = (sim_diff < threshold).sum().type(torch.Tensor)
        false_p = (sim_diff >= threshold).sum().type(torch.Tensor)
        FP_acc.append(true_p / len(sim_same) / 2 + true_n / len(sim_diff) / 2)
        FAR.append(false_p / len(sim_diff))
        FRR.append(false_n / len(sim_same))
        TAR.append(true_p / len(sim_same))
    FAR = torch.stack(FAR)
    FRR = torch.stack(FRR)
    TAR = torch.stack(TAR)
    FP_acc = torch.stack(FP_acc)
    acc = FP_acc.max()
    delta = (FAR - FRR).abs()
    EER_idx = delta.argmin()
    EER = FAR[EER_idx] / 2 + FRR[EER_idx] / 2
    sum_y = TAR[:-1] + TAR[1:]
    delta_x = FAR[:-1] - FAR[1:]
    AUC = sum_y.dot(delta_x) / 2
    return acc, EER, AUC


def find_wrong_match(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    #找出错误的配对
    """
    find the probe samples which result in a wrong match at rank-1.
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    prb_idx: list of int, length == n_found_wrong_prb
    gal_idx: list of np array, each of which associating with the element in prb_idx
    correct_indicators: list of np array corresponding to gal_idx, indicating whether that gal is a correct match.
    """
    is_view_sensitive = False
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        is_view_sensitive = True
    prb_idx = []
    gal_idx = []
    correct_indicators = []

    for i in range(num_probe):
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        if pos_first_correct != 0:
            prb_idx.append(i)#记录fank-1配对不正确的query
            gal_idx.append(ranking_list)#该query对应的gallery的索引
            correct_indicators.append(probe_label == inference_list)#该query对应的gallery的feature

    return prb_idx, gal_idx, correct_indicators


def plot_ranking_imgs(gal_dataset, prb_dataset, gal_idx, prb_idx, n_gal=8, size=(256, 128), save_path='',
                      correct_indicators=None, sample_prb=False, n_prb=8):
    #用于展示匹配结果的一个函数
    """
    plot ranking imgs and save it.
    :param gal_dataset: should support indexing and return a tuple, in which the first element is an img,
           represented as np array  利用索引，需要找到对应的图片
    :param prb_dataset:
    :param gal_idx: list of np.array, each of which corresponds to the element in prb_idx
    :param prb_idx: list of int, indexing the prb_dataset
    :param n_gal: number of gallery imgs shown in a row (for a probe).#查询距离最近的前几张图片
    :param size: resize all shown imgs#展示的图片大小固定
    :param save_path: directory to save; the file name is ranking_(time string).png#保存的图片路径
    :param correct_indicators: list of np array corresponding to gal_idx, indicating whether that
           gal is a correct match. if specified, each correct match will has a small green box in the upper-left.#记录配对序列的是否正确
    :param sample_prb: if True, the prb_idx is randomly sampled n_prb samples; otherwise, keep the order of prb_idx#是否随机突出query_set
    and plot all the images specified in prb_idx.
    :param n_prb: if sample_prb is True, we sample n_prb probe images.
    :return:
    """
    assert len(prb_idx) == len(gal_idx)
    if correct_indicators is not None:
        assert len(prb_idx) == len(correct_indicators)
    box_size = tuple(map(lambda x: int(x/12.0), size))

    is_gal_on = gal_dataset.on_transform#数据裁剪处理
    is_prb_on = prb_dataset.on_transform
    gal_dataset.turn_off_transform()
    prb_dataset.turn_off_transform()

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    n_prb = len(prb_idx) if n_prb > len(prb_idx) else n_prb
    if correct_indicators is None:
        if sample_prb:
            used = sample(list(zip(prb_idx, gal_idx)), n_prb)
        else:
            used = list(zip(prb_idx, gal_idx))
        imgs = []
        for p_idx, g_idx_array in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx in g_idx_array[:n_gal_used]:
                gal_img = transform(gal_dataset[g_idx][0])
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))
    else:
        if sample_prb:
            used = sample(list(zip(prb_idx, gal_idx, correct_indicators)), n_prb)
        else:
            used = list(zip(prb_idx, gal_idx, correct_indicators))
        imgs = []
        for p_idx, g_idx_array, correct_ind in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx, is_correct_match in zip(g_idx_array[:n_gal_used], correct_ind[:n_gal_used]):
                gal_img = transform(gal_dataset[g_idx][0])
                if is_correct_match:
                    gal_img[0, :box_size[0], :box_size[1]].zero_()
                    gal_img[1, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                else:
                    gal_img[0, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[1, :box_size[0], :box_size[1]].zero_()
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))#输出为形状和x一致的矩阵，其元素全部为0

    filename = os.path.join(save_path, 'ranking_{}.png'.format(time_string()))
    torchvision.utils.save_image(imgs, filename, nrow=n_gal+1)#保存匹配正确的图片
    print('saved ranking images into {}'.format(filename))
    gal_dataset.on_transform = is_gal_on
    prb_dataset.on_transform = is_prb_on


def parse_pretrained_checkpoint(checkpoint, num_classes, fc_layer_name='fc', is_preserve_fc=False):
    #参数预训练模型，关于fc层要不要，这里有一个选择
    """
    :param checkpoint: OrderedDict (state_dict) or a tuple (checkpoint)
    :param num_classes:
    :param fc_layer_name:
    :param is_preserve_fc: if True, when fc layer output dimension is different from num_classes,
    we still preserve the parameters by removing additional dim or adding randomly init missing dim.
    :return: state_dict: a state dict whose fc layer is processed,
    i.e. if the fc output is not num_classes, remove the fc weight and fc bias (if exists)
    if is_preserve_fc=False; otherwise preserve the weights by adding or removing dimensions.
    """
    if isinstance(checkpoint, OrderedDict):
        print('loaded a state dict.')
        state_dict = checkpoint
    elif isinstance(checkpoint, tuple):
        print('loaded a checkpoint.')
        net = checkpoint[0].net
        if isinstance(net, torch.nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
    else:
        assert False, 'unknown type {}\n'.format(type(checkpoint))
    fc_weight_name = '{}.weight'.format(fc_layer_name)
    try:
        fc_weight = state_dict[fc_weight_name]
        output_dim = fc_weight.shape[0]
        if output_dim != num_classes:
            print('The output dim not match the specified num_classes. fc param is removed.\n')
            state_dict.pop(fc_weight_name)
            if '{}.bias'.format(fc_layer_name) in state_dict:
                state_dict.pop('{}.bias'.format(fc_layer_name))
    except KeyError:
        print('parse_pretrained_checkpoint: No fc weights found.')
    return state_dict  #以数据字典的形式返回模型的参数

#这两个不看先
def pair_idx_to_dist_idx(d, i, j):
    """
    :param d: numer of elements
    :param i: np.array. i < j in every element
    :param j: np.array
    :return:
    """
    assert np.sum(i < j) == len(i)
    index = d*i - i*(i+1)/2 + j - 1 - i
    return index.astype(int)


def dist_idx_to_pair_idx(d, i):
    """
    :param d: number of samples
    :param i: np.array
    :return:
    """
    if i.size == 0:
        return None
    b = 1 - 2*d
    x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y
#这两个不看先

def int2onehot(labels, num_classes):
    #onehot编码
    """
    :param labels: torch.Tensor, shape=(BS,)
    :param num_classes: int, the dimension of returned label vectors
    :return:
    """
    batch_size = labels.shape[0]
    onehot_labels = torch.zeros(batch_size, num_classes).cuda()
    idx = torch.arange(batch_size)
    onehot_labels[idx, labels] = 1
    return onehot_labels


def test():
    labels = torch.tensor([1,2,3,4,5,6]).cuda()
    onehot = int2onehot(labels, 10)
    print(onehot)#onehot编码
    pass


if __name__ == '__main__':
    test()
