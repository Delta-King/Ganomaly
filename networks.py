import torch
import torch.nn as nn
import torch.nn.parallel


def weights_init(mod):
    """权重的初始化 Custom weights initialization called on netG, netD"""
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:  # 这里的Conv和BatchNorm是torch.nn里的形式
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)  # bn层里初始化γ, 服从(1,0.02)的正态分布
        mod.bias.data.fill_(0)  # bn层里初始化β, 默认为0


class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        """
        编码器 DCGAN ENCODER NETWORK
        :param isize: input image size
        :param nz: in netG: size of the latent z vector, default 100; in netD: 1, because output needs to be 1 x 1 x 1
        :param nc: input image channels, default 3
        :param ndf: encoder network channels used
        :param ngpu: number of GPUs to use
        :param n_extra_layers: Number of extra layers on generator and discriminator
        :param add_final_conv:
        """
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # nn.Conv2d: in_channels, out_channels, kernel_size, stride, padding
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        # layer output is ndf x (isize/2) x (isize/2)
        main.add_module('initial-relu-{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        for t in range(n_extra_layers):
            # layer input is cndf x csize x csize
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf), nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf), nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf), nn.LeakyReLU(0.2, inplace=True))
            # layer output is cndf x csize x csize

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        # layer output is final_cndf x 4 x 4, final_cndf = original_cndf x (log2(csize)-2)

        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1), nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        # output is nz x 1 x 1

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Decoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        """
        解码器 DCGAN DECODER NETWORK
        :param isize: input image size
        :param nz: size of the latent z vector, default 100
        :param nc: input image channels, default 3
        :param ngf: decoder network channels used
        :param ngpu: number of GPUs to use
        :param n_extra_layers: Number of extra layers on generator and discriminator
        """
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        # 此时cngf = ngf // 2 * isize / 4

        main = nn.Sequential()
        # input is nz x 1 x 1
        # nn.ConvTranspose2d: in_channels, out_channels, kernel_size, stride, padding
        main.add_module('initial-{0}-{1}-conv'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        # layer output is cngf x 4 x 4
        main.add_module('initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))
        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-conv'.format(cngf, cngf // 2), nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf), nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf), nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf), nn.ReLU(True))
        main.add_module('final-{0}-{1}-conv'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc), nn.Tanh())

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class NetD(nn.Module):
    """定义判别器（编码器）"""
    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        # list is not a Module subclass; Module.children is not a Module subclass 这里的*就起了作用，将list或者children的内容迭代的一个一个的传进去

        self.classifier = nn.Sequential(layers[-1])

        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)  # 此时的结果是1 x 1 x 1, 值在[0,1]
        classifier = classifier.view(-1, 1).squeeze(1)  # 将shape(1,1,1)变为shape(1)

        return classifier, features


class NetG(nn.Module):
    """定义生成器（编码器+解码器+编码器）"""
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)  # 100 x 1 x 1
        gen_img = self.decoder(latent_i)  # nc x isize x isize
        latent_o = self.encoder2(gen_img)  # 100 x 1 x 1

        return gen_img, latent_i, latent_o


def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))


def l2_loss(input, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)
