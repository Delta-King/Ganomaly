import argparse
import os
import torch


class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base
        self.parser.add_argument('--dataset', default='', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
        self.parser.add_argument('--isize', type=int, default=128, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=2, help='Number of extra layers on generator and discriminator')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='dalishi_new', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--output_folder', default='output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manual_seed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')

        # Train
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--print_freq', type=int, default=128, help='frequency of showing training results on console')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--save_image_freq', type=int, default=4480, help='save image freq')  # 4480
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Adversarial loss weight')
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.isTrain = True
        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                self.opt.gpu_ids.append(gpu_id)
        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])
        args = vars(self.opt)

        # save to the disk
        expr_dir = os.path.join(self.opt.output_folder, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.output_folder, self.opt.name, 'test')
        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
