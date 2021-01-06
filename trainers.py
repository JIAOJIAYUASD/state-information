from sklearn.cluster import KMeans#K-means聚类
from resnet import *
from utils import *
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from scipy.spatial.distance import cdist
from collections import Counter


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:#属性名字
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()
            #将names中的给定属性设置为training状态

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()
            # 将names中的给定属性设置为evaluation状态


class ReidTrainer(Trainer):
    #从上面的Trainer继承而来的
    def __init__(self, args, logger, loader):
        super(ReidTrainer, self).__init__()
        self.args = args
        self.logger = logger

        self.disc_loss = nn.CrossEntropyLoss().cuda()#交叉熵损失函数
        self.align_loss = AlignLoss(args.batch_size).cuda()#一个不知名的loss
        
        self.net = resnet50(pretrained=False, num_classes=args.pseudo_class).cuda()#继承resnet50，默认是不预训练
        if args.pretrain_path is None:
            self.logger.print_log('do not use pre-trained model. train from scratch.')#没找到预训练模型的位置
        elif os.path.isfile(args.pretrain_path):#load pre-trained model
            checkpoint = torch.load(args.pretrain_path)
            state_dict = parse_pretrained_checkpoint(checkpoint, args.pseudo_class)#返回预训练模型的参数
            state_dict = self.add_fc_dim(state_dict, loader)#添加fc层参数
            self.net.load_state_dict(state_dict, strict=False)#将参数放到模型里面
            self.logger.print_log('loaded pre-trained model from {}'.format(args.pretrain_path))
        else:
            self.logger.print_log('{} is not a file. train from scratch.'.format(args.pretrain_path))
        self.net = nn.DataParallel(self.net).cuda()#DataParallel并行计算只存在在前向传播

        bn_params, other_params = partition_params(self.net, 'bn')#将BatchNorm层的参数与其他层的参数分开
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                          {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)#weight_decay权值l2正则化下，衰减方式
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(args.epochs/8*5), int(args.epochs/8*7)])#多阶段训练，后面的学习率会降下来的

        self.init_losses(loader)
        self.prior = torch.ones(args.pseudo_class).cuda()
        self.n_total = torch.full((args.pseudo_class,), len(loader.dataset)/args.pseudo_class)#生成是一阶张量，所有元素均为同一个值
        self.max_predominance_index = torch.zeros(args.pseudo_class)#生成一阶0张量，用于记录每一个代理类中数目最多的状态的个数
        self.pseudo_label_memory = torch.full((len(loader.dataset),), -1, dtype=torch.long)#生成一阶张量，所有元素都是-1
        self.view_memory = np.asarray(loader.dataset.views)#状态信息这个tensor转np数组
        #以上几个变量，后面都会进行更新，但是这里为什么要更新啊？而且还是以动量形式更新

    def train_epoch(self, loader, epoch):
        self.lr_scheduler.step()#开始前馈训练
        batch_time_meter = AverageMeter()
        stats = ('loss_surrogate', 'loss_align', 'loss_total')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()

        end = time.time()
        for i, tuple in enumerate(loader):
            if i % self.args.prior_update_freq == 0:#每50次更新一次
                self.update_state()#更新
                self.update_prior()
            imgs = tuple[0].cuda()
            views = tuple[2].cuda()
            idx_img = tuple[3]

            classifer = self.net.module.fc.weight.renorm(2, 0, 1e-5).mul(1e5)#FC层的权值归一化

            features, similarity, _ = self.net(imgs)
            scores = similarity * 30
            logits = F.softmax(features.mm(classifer.detach().t() * 30), dim=1)#features与classifer求内积（内积大小作为预测的分数），且对classifer可返回梯度
            loss_align = self.align_loss(features, views)#计算Ldift，根据view下的分布，计算对应的mean和std

            pseudo_labels = get_pseudo_labels(logits.detach()*self.prior)#根据最大的内积和pk,计算预测的伪类标签
            self.pseudo_label_memory[idx_img] = pseudo_labels.cpu()#放进这个list中，图片对应伪类标签
            loss_surrogate = self.disc_loss(scores, pseudo_labels)#计算伪类损失Lsurr，为什么啊，这个是用scores和pseudo_labels来计算Lsurr，而不是用feature和聚类中心的

            self.optimizer.zero_grad()#梯度清零
            loss_total = loss_surrogate + self.args.lamb_align * loss_align#这个就是模型的最终损失函数，lamb_align是正则化系数
            loss_total.backward()#梯度回传
            self.optimizer.step()

            for k in stats:
                v = locals()[k]
                if v.item() > 0:
                    meters_trn[k].update(v.item(), self.args.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if self.args.print_freq != 0 and i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(loader), freq) + create_stat_string(meters_trn) + time_string())

        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, loader, gallery_loader, probe_loader):
        stats = ('r1', 'r5', 'r10', 'MAP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')#计算余弦相似性
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views, ignore_MAP=False)
        r1 = CMC[0]
        r5 = CMC[4]
        r10 = CMC[9]

        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item(), self.args.batch_size)
        return meters_val

    def eval_performance_mpie(self, target_loader, gallery_loader, probe_loader):
        stats = ('overall', 'd0', 'd15', 'd30', 'd45', 'd60')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        accuracy = []
        for v in np.unique(probe_views):
            idx = probe_views == v
            f = probe_features[idx]
            l = probe_labels[idx]
            dist = cdist(gallery_features, f, metric='cosine')
            acc = eval_acc(dist, gallery_labels, l)
            accuracy.append(acc)
            self.logger.print_log('view : {}, acc : {:2f}'.format(v, acc))
        accuracy = np.array(accuracy)
        overall = accuracy.mean()
        d0 = accuracy[4]
        d15 = (accuracy[3]+accuracy[5])/2
        d30 = (accuracy[2]+accuracy[6])/2
        d45 = (accuracy[1]+accuracy[7])/2
        d60 = (accuracy[0] + accuracy[8]) / 2
        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item(), self.args.batch_size)
        return meters_val

    def eval_performance_cfp(self, test_loader, protocol):
        stats = ('acc_FP', 'EER_FP', 'AUC_FP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        test_features, _, _ = extract_features(test_loader, self.net, index_feature=0, return_numpy=False)
        similarity = (test_features.matmul(test_features.t())+1)/2
        FP_same_idx = protocol['FP_same']
        FP_same = []
        for pair in FP_same_idx:
            sim = similarity[pair[0], pair[1]+500]
            FP_same.append(sim)
        FP_same = torch.stack(FP_same).cpu()
        FP_diff_idx = protocol['FP_diff']
        FP_diff = []
        for pair in FP_diff_idx:
            sim = similarity[pair[0], pair[1]+500]
            FP_diff.append(sim)
        FP_diff = torch.stack(FP_diff).cpu()
        acc_FP, EER_FP, AUC_FP = eval_CFP(FP_same, FP_diff)

        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item())
        return meters_val

    def init_losses(self, loader):
        if os.path.isfile(self.args.align_path):
            features, views = torch.load(self.args.align_path)
            self.logger.print_log('loaded features from {}'.format(self.args.align_path))
        else:
            self.logger.print_log('not found {}. computing features...'.format(self.args.align_path))
            features, _, views = extract_features(loader, self.net, index_feature=0, return_numpy=False)
            torch.save((features, views), self.args.align_path)
        self.align_loss.init_centers(features, views)#为什么这里就是全部特征及其状态信息啊
        self.logger.print_log('initializing align loss centers done.')

    def add_fc_dim(self, state_dict, loader, fc_layer_name='fc'):#加上fc层的参数
        fc_weight_name = '{}.weight'.format(fc_layer_name)
        fc_weight = state_dict[fc_weight_name] if fc_weight_name in state_dict else torch.empty(0, 2048).cuda()#接口是2048维矩阵(空的)
        if os.path.isfile(self.args.centroids_path):#观察是否有提前准备好聚类的中心，如果无，则自己训练
            renorm_centroids = torch.load(self.args.centroids_path)#加载模型
            self.logger.print_log('loaded centroids from {}.'.format(self.args.centroids_path))
        else:#无找到训练好的聚类中心时，需要重新通过Kmeans来训练
            self.logger.print_log('Not found {}. Evaluating centroids ..'.format(self.args.centroids_path))
            self.net.load_state_dict(state_dict, strict=False)#这些是fc层前面的参数
            self.eval()

            features, _, _ = extract_features(loader, self.net, index_feature=0)
            kmeans = KMeans(n_clusters=self.args.pseudo_class, n_init=2)
            '''
            n_clusters:聚类中心数目
            n_init:用不同的质心初始化值运行算法的次数
            '''
            kmeans.fit(features)#计算k-means聚类
            centroids_np = kmeans.cluster_centers_#向量，[n_clusters, dim_features] (聚类中心，向量表示)
            centroids = torch.Tensor(centroids_np).cuda()#放到GPU

            fc_weights = self.net.fc.weight.data
            mean_norm = fc_weights.pow(2).sum(dim=1).pow(0.5).mean()#按列求和
            renorm_centroids = centroids.renorm(p=2, dim=0, maxnorm=(1e-5) * mean_norm).mul(1e5)#归一化
            torch.save(renorm_centroids, self.args.centroids_path)#保存聚类中心的特征
        new_fc_weight = torch.cat([fc_weight, renorm_centroids], dim=0)#按维数0（行）拼接，张量拼接（这种情况下，fc_weight这个矩阵是，所以这里有一点像list的append）
        state_dict[fc_weight_name] = new_fc_weight
        self.logger.print_log('FC dimensions added.')#FC层的参数是有聚类中心产生的
        return state_dict

    def update_state(self, moment=0.5):
        """
        according to self.pseudo_label_memory and self.view_memory,
        update self.n_total and self.max_predominance_index
        :return:
        """
        n_total = torch.zeros(self.args.pseudo_class)#代理类的数目
        max_predominance_index = torch.ones(self.args.pseudo_class)
        for i in range(self.args.pseudo_class):
            idx = self.pseudo_label_memory == i
            n_total[i] = idx.sum()
            views = self.view_memory[idx.nonzero().squeeze(dim=1).numpy()]
            t = tuple(views)
            if t:
                c = Counter(t)
                _, max_count = c.most_common(1)[0]#找到数目最多的状态信息
                max_predominance_index[i] = max_count/n_total[i]#计算Rk
        self.n_total = moment * self.n_total + (1-moment) * n_total#这里动量更新每个代理类的样本数目和其对应的Rk
        self.max_predominance_index = moment * self.max_predominance_index + (1-moment) * max_predominance_index

    def update_prior(self):
        for i in range(self.args.pseudo_class):
            a = self.args.a
            b = self.args.b
            x = self.max_predominance_index[i]
            self.prior[i] = 1/(1+np.exp(a*(x-b)))#计算p(k)，根据a的大小区分为软和硬


class AlignLoss(torch.nn.Module):#这部分应该和simplified 2-Wasserstein distance相关
    def __init__(self, batch_size):
        super(AlignLoss, self).__init__()
        self.moment = batch_size / 10000
        self.initialized = False

    def init_centers(self, variables, views):#传入特征向量和状态信息
        """
        :param variables: shape=(N, n_class)
        :param views: (N,)
        :return:
        """
        #计算每个状态信息的特征向量的均值和方差
        univiews = torch.unique(views)#状态信息唯一化
        mean_ml = []
        std_ml = []
        for v in univiews:
            #每个状态信息下的子分布，计算其均值和方差，为了得到simplified 2-Wasserstein distance
            ml_in_v = variables[views == v]#这里应该是按不同的状态信息，求对应的均值和方差
            mean = ml_in_v.mean(dim=0)#按行操作
            std = ml_in_v.std(dim=0)
            mean_ml.append(mean)#记录起来
            std_ml.append(std)
        center_mean = torch.mean(torch.stack(mean_ml), dim=0)#把所有记录的均值和方差求均值
        center_std = torch.mean(torch.stack(std_ml), dim=0)
        self.register_buffer('center_mean', center_mean)#在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出。
        self.register_buffer('center_std', center_std)
        self.initialized = True

    def _update_centers(self, variables, views):
        """
        #更新每个状态信息的特征向量的均值和方差
        :param variables: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        univiews = torch.unique(views)#状态信息唯一化
        means = []
        stds = []
        for v in univiews:
            ml_in_v = variables[views == v]#这里应该是按不同的状态信息，求对应的均值和方差
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            std = ml_in_v.std(dim=0)
            stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean*(1-self.moment) + new_mean*self.moment#动量更新均值和方差，算法中的15行
        new_std = torch.mean(torch.stack(stds), dim=0)
        self.center_std = self.center_std*(1-self.moment) + new_std*self.moment

    def forward(self, variables, views):
        """
        计算损失
        :param variables: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        self._update_centers(variables.detach(), views)#通过.detach() “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的

        univiews = torch.unique(views)
        loss_terms = []
        for v in univiews:
            ml_in_v = variables[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()#计算损失
            loss_terms.append(loss_mean)
            std = ml_in_v.std(dim=0)
            loss_std = (std - self.center_std).pow(2).sum()
            loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total#返回的是Ldift


def get_pseudo_labels(similarity):
    """
    获得伪类标签
    :param similarity: torch.Tensor, shape=(BS, n_classes)
    :return:
    """
    sim = similarity
    max_entries = torch.argmax(sim, dim=1)
    pseudo_labels = max_entries
    return pseudo_labels.cuda()
