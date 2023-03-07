from collections import OrderedDict
import os
import time
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from evaluate import evaluate
from networks import NetD, NetG, weights_init, l2_loss

from visualizer import Visualizer


class Ganomaly:

    @property  # @property装饰器会将方法转换为相同名称的只读属性
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, dataloader):  # __init__是类的构造函数，在类的实例创建后被立即调用，可以在__init__中将变量赋值给class自己的属性变量
        super(Ganomaly, self).__init__()

        # Initialize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.output_folder, self.opt.name, 'train')  # 训练结果保存的路径
        self.tst_dir = os.path.join(self.opt.output_folder, self.opt.name, 'test')  # 测试结果保存的路径
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")  # cuda:0 表示是从第0块gpu开始

        # 生成器的参数
        self.fake = None
        self.latent_i = None
        self.latent_o = None
        self.err_g_adv = None
        self.err_g_con = None
        self.err_g_enc = None
        self.err_g = None

        # 判别器的参数
        self.pred_real = None
        self.feat_real = None
        self.pred_fake = None
        self.feat_fake = None
        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        # Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        self.gt_labels = None
        self.an_scores = None

        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        # 继续训练时调用
        if self.opt.resume != '':  # resume是上一次模型训练的结果所保存的地址
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        # 损失函数
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()  # 交叉熵损失函数，用于判别器

        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        # Setup optimizer.
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def update_netg(self):
        # Forward propagate through netG
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

        # Backward propagate through netG
        self.optimizer_g.zero_grad()
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])  # 计算生成器生成的假图和原始输入的真图通过判别器得到的特征图之间的距离
        self.err_g_con = self.l_con(self.fake, self.input)  # 计算输入图片与生成器生成的图片之间的距离
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)  # 计算两个下采样之后的向量的距离
        self.err_g = self.err_g_adv * self.opt.w_adv + self.err_g_con * self.opt.w_con + self.err_g_enc * self.opt.w_enc  # 权重比原论文建议为1:50:1
        self.err_g.backward(retain_graph=True)
        self.optimizer_g.step()

    def update_netd(self):
        # Forward propagate through netD
        self.pred_real, self.feat_real = self.netd(self.input)  # 输入真实图片经过判别器得到的[0,1]之间的分数和特征图
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())  # 生成的假图经过判别器得到的[0,1]之间的分数和特征图 .detach()消除梯度属性

        # Backward propagate through netD
        self.optimizer_d.zero_grad()
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
        self.optimizer_d.step()

    def reinitialize_netd(self):
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize(self):
        """前向传播、损失计算、反向传播，更新网络D、网络G"""

        # 需要先前向传播netG以生成假图
        self.update_netg()
        self.update_netd()

        # 如果判别器的loss=0,重新初始化判别器的参数
        if self.err_d.item() < 1e-5:
            self.reinitialize_netd()

    def get_errors(self):
        """ Get netD and netG errors.
        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([('err_d', self.err_d.item()),  # 判别器的loss
                              ('err_g', self.err_g.item()),  # 生成器的loss
                              ('err_g_adv', self.err_g_adv.item()),  # 生成器生成的图片通过判别器得到的分数与真实标签1之间的距离
                              ('err_g_con', self.err_g_con.item()),  # 输入图片与生成器生成的图片之间的距离
                              ('err_g_enc', self.err_g_enc.item())])  # 两个下采样之后的向量的距离
        return errors

    # 获得输入图片、输入图片经过生成器生成的图片、残差图片
    def get_current_images(self):
        reals = self.input.data
        fakes = self.fake.data
        residuals = abs(reals - fakes)
        residuals_gray = transforms.Grayscale(num_output_channels=1)(residuals)

        def binary(x, y):
            return 0 if x < 0.5 else 1

        residuals_binary = residuals_gray.cpu().map_(residuals_gray.cpu(), binary)

        return reals, fakes, residuals, residuals_gray, residuals_binary

    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.
        Args:
            epoch ([int]): Current epoch number.
        """
        weight_dir = os.path.join(self.opt.output_folder, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()}, '%s/netG.pth' % weight_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()}, '%s/netD.pth' % weight_dir)

    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth
        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])  # 把data的第一项：图片数据复制给self.input
            self.gt.resize_(input[1].size()).copy_(input[1])  # 把data的第二项：图片的标签复制给self.gt
            self.label.resize_(input[1].size())

    def train_one_epoch(self):
        self.netg.train()  # 当网络中有dropout和bn时, 训练要net.train(), 测试要net.eval()
        epoch_iter = 0
        # dataloader出来有2个参数，第一个是数据，第二个是标签
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize()

            if self.total_steps % self.opt.print_freq == 0:  # 每print_freq的倍数次在console展示训练结果
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:  # 每save_image_freq的倍数次 保存模型生成的图片
                save_index = self.total_steps / self.opt.save_image_freq
                reals, fakes, residuals, residuals_gray, residuals_binary = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, residuals, residuals_gray, residuals_binary, save_index, 'train')
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, residuals, residuals_gray, residuals_binary)

        print(">> Training model %s. Epoch %d/%d Loss_D: %.3f Loss_G %.3f " % (self.name, self.epoch + 1, self.opt.niter, self.err_d.item(), self.err_g.item()))

    # 训练模型
    def train(self):
        self.total_steps = 0
        best_auc = 0

        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print('>> Training model %s.[Done]' % self.name)

    # 测试模型
    def test(self):
        with torch.no_grad():  # 做异常检测的时候分数只由生成器来决定，所以不调用判别器的参数
            if self.opt.load_weights:
                path = os.path.join(self.opt.output_folder, self.opt.name, 'train', 'weights', 'netG.pth')
                pretrained_dict = torch.load(path)['state_dict']
                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'
            # an_scores最终给每一张图片一个[0,1]之间的数，通过比较这个数和阈值之间的大小，判断这张图是否正常
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            # gt_labels是测试集里打的真实的标签
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
            # self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            # self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            # 因为dataloader是个迭代器，所以用enumerate，且后面的那个0表示i从0开始表示，如果后面那个是1，那么i就从1开始表示
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)  # 异常分即两个特征向量之间的平均距离 error的size是(batchsize,1,1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))  # (,batchsize)
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))  # (,batchsize)
                # self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)  # (batchsize,nz)
                # self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)  # (batchsize,nz)
                self.times.append(time_o - time_i)

                # 保存图片
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.output_folder, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    reals, fakes, residuals, residuals_gray, residuals_binary = self.get_current_images()
                    self.visualizer.save_current_images(self.epoch, reals, fakes, residuals, residuals_gray, residuals_binary, i + 1, 'test')

            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)  # metric默认是auc
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            return performance

    # 在测试集上选取阈值
    def select_threshold(self):
        with torch.no_grad():
            path = os.path.join(self.opt.output_folder, self.opt.name, 'train', 'weights', 'netG.pth')
            pretrained_dict = torch.load(path)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")
            print('   Loaded weights.')
            self.netg.eval()  # model.eval()的作用是不启用 Batch Normalization 和 Dropout. 保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变；不进行随机舍弃神经元

            best_correct = 0
            best_e = 0
            best_accuracy = 0
            for e in np.arange(0.001, 0.02, 0.001):
                self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
                self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
                self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
                self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
                for i, data in enumerate(self.dataloader['test'], 0):
                    self.set_input(data)
                    self.fake, latent_i, latent_o = self.netg(self.input)  # 只用到 latent_i, latent_o
                    # error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                    residual_binary = self.get_current_images()[4]
                    error = torch.mean(residual_binary.view(self.fake.size(0), -1), dim=1)
                    self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                    self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                    # self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                    # self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)
                # self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
                self.an_scores[self.an_scores >= e] = 1
                self.an_scores[self.an_scores < e] = 0
                correct_normal = ((self.an_scores.int() == self.gt_labels.int()) & (self.gt_labels.int() == 0)).sum().item()
                correct_abnormal = ((self.an_scores.int() == self.gt_labels.int()) & (self.gt_labels.int() == 1)).sum().item()
                correct = correct_normal + correct_abnormal
                total_normal = (self.gt_labels.int() == 0).sum().item()
                total_abnormal = (self.gt_labels.int() == 1).sum().item()
                total = total_normal + total_abnormal
                if correct > best_correct:
                    best_correct = correct
                    best_e = e
                    best_accuracy = round(100 * correct / total, 2)
                print('Test Accuracy of the model on the {} test images: {} %, normal accuracy: {} %, abnormal accuracy: {} %, e: {}'.format(
                    total, round(100 * correct / total, 2), round(100 * correct_normal / total_normal, 2), round(100 * correct_abnormal / total_abnormal, 2), round(e, 3)))
            print('Best e: {}, the accuracy is {} %'.format(round(best_e, 3), best_accuracy))

    # 用选好的阈值在验证集上看看效果
    def validate(self, threshold_value):
        with torch.no_grad():
            path = os.path.join(self.opt.output_folder, self.opt.name, 'train', 'weights', 'netG.pth')
            pretrained_dict = torch.load(path)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")
            print('   Loaded weights.')
            self.netg.eval()

            correct = 0
            total = 0

            self.an_scores = torch.zeros(size=(len(self.dataloader['validate'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['validate'].dataset),), dtype=torch.long, device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['validate'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['validate'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            for i, data in enumerate(self.dataloader['validate'], 0):
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            self.an_scores[self.an_scores >= threshold_value] = 1
            self.an_scores[self.an_scores < threshold_value] = 0
            correct += (self.an_scores.float() == self.gt_labels.float()).sum().item()
            total += self.gt_labels.size(0)

            print('Test Accuracy of the model on the {} validate images: {}% '.format(total, round(100 * correct / total, 2)))
            print('correct:%d; total:%d' % (correct, total))
