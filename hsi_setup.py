import os
from os.path import join

import torch
import torch.optim as optim

import models
from utility import *


class Engine(object):
    def __init__(self, prefix, opt):
        self.prefix = prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.gpu_ids = self.opt.gpu_ids

        self.__setup()


    def __setup(self):
        self.basedir = join('checkpoint', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        print('Cuda Acess: %d' %cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        # print(self.net)

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)

        self.criterion = nn.MSELoss()

        if cuda:
            self.net.cuda()

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(self.opt.arch)

        """Optimization Setup"""
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)        

        """Resume previous model"""
        if self.opt.resume:
            self.load(self.opt.resumePath)
        else:
            print('==> Building model..')


    def __step(self, train, inputs, targets):
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        bandwise = self.net.module.bandwise if len(self.opt.gpu_ids) > 1 else self.net.bandwise
        if bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1,1), targets.split(1,1))):                
                o = self.net(i)                
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            self.optimizer.step()
        return outputs, loss_data


    def load(self, resumePath=None):
        model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_psnr = best_model['psnr']
            self.best_loss = best_model['loss']

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        
    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs, loss_data = self.__step(True, inputs, targets)

            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(join(self.prefix,'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(join(self.prefix,'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e'
                % (avg_loss, loss_data))

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix,'train_loss_epoch'), avg_loss, self.epoch)


    """Validation"""
    def validate(self, valid_loader):        
        self.net.eval()
        validate_loss = 0
        total_psnr = 0

        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs, loss_data = self.__step(False, inputs, targets)

            bwloss_data = bwmse_loss(outputs, targets)
            psnr = mpsnr(bwloss_data)

            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx+1)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx+1)


            progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                % (avg_loss, avg_psnr))
        
        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix,'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix,'val_psnr_epoch'), avg_psnr, self.epoch)

        """Save checkpoint"""
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
            self.save_checkpoint(psnr=avg_psnr, loss=avg_loss, model_out_path=model_best_path)
            self.best_psnr = avg_psnr
            self.best_loss = avg_loss
        return avg_psnr, avg_loss


    """Testing"""
    def test(self, test_loader, name):
        self.net.eval()
        test_loss = 0
        total_psnr = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs, loss_data = self.__step(False, inputs, targets)

            test_loss += loss_data
            avg_loss = test_loss / (batch_idx+1)

            bwloss_data = bwmse_loss(outputs, targets)
            psnr = mpsnr(bwloss_data)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx+1)


            progress_bar(batch_idx, len(test_loader), 'Loss: %.4e | PSNR: %.4f'
                % (avg_loss, avg_psnr))

        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix, name, 'test_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, name, 'test_psnr_epoch'), avg_psnr, self.epoch)
        
        return avg_psnr


    def save_checkpoint(self, psnr, loss, model_out_path=None):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" %(self.epoch, self.iteration))

        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'psnr': psnr,
            'loss': loss,
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.mkdir(join(self.basedir, self.prefix))
        
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

