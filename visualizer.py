import os
import time
import numpy as np
import torchvision.utils as vutils


class Visualizer:

    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(env='train', server=opt.display_server, port=opt.display_port)
        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None
        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.output_folder, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.output_folder, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.output_folder, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    ##

    @staticmethod
    def normalize(inp):
        """Normalize the tensor
        Args:
            inp ([FloatTensor]): Input tensor
        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)
    ##

    def plot_current_errors(self, epoch, counter_ratio, errors):

        """Plot current errors.
        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win='current errors'
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance
        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win='performance'
        )
    ##

    def print_current_errors(self, epoch, errors):
        """ Print current errors.
        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    ##

    def print_current_performance(self, performance, best):
        """ Print current performance results.
        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max AUC: %.3f' % best
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_current_images(self, reals, fakes, residuals, residuals_gray, residuals_binary):
        """ Display current images.
        Args:
            reals ([FloatTensor])           : Real Image
            fakes ([FloatTensor])           : Fake Image
            residuals ([FloatTensor])       : Residual Image
            residuals_gray ([FloatTensor])  : Residual Gray Image
            residuals_binary ([FloatTensor]): Residual Binary Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        residuals = self.normalize(residuals.cpu().numpy())
        residuals_gray = self.normalize(residuals_gray.cpu().numpy())
        residuals_binary = self.normalize(residuals_binary.cpu().numpy())
        self.vis.images(reals, win='reals', opts={'title': 'Reals'})
        self.vis.images(fakes, win='fakes', opts={'title': 'Fakes'})
        self.vis.images(residuals, win='residuals', opts={'title': 'Residuals'})
        self.vis.images(residuals_gray, win='residuals_gray', opts={'title': 'Residuals Gray'})
        self.vis.images(residuals_binary, win='residuals_binary', opts={'title': 'Residuals Binary'})

    def save_current_images(self, epoch, reals, fakes, residuals, residuals_gray, residuals_binary, save_index, phase):
        """ Save images for epoch i.
        Args:
            save_index                      : Save index
            epoch ([int])                   : Current epoch
            reals ([FloatTensor])           : Real Image
            fakes ([FloatTensor])           : Fake Image
            residuals ([FloatTensor])       : Residual Image
            residuals_gray ([FloatTensor])  : Residual Gray Image
            residuals_binary ([FloatTensor]): Residual Binary Image
            phase ([string])                : For train or test
        """
        dst = self.tst_img_dir if phase == 'test' else self.img_dir
        vutils.save_image(reals, '%s/%03d_%01d_reals.png' % (dst, epoch + 1, save_index), normalize=True)
        vutils.save_image(fakes, '%s/%03d_%01d_fakes.png' % (dst, epoch + 1, save_index), normalize=True)
        vutils.save_image(residuals, '%s/%03d_%01d_residuals.png' % (dst, epoch + 1, save_index), normalize=True)
        vutils.save_image(residuals_gray, '%s/%03d_%01d_residuals_gray.png' % (dst, epoch + 1, save_index), normalize=True)
        vutils.save_image(residuals_binary, '%s/%03d_%01d_residuals_binary.png' % (dst, epoch + 1, save_index), normalize=True)
