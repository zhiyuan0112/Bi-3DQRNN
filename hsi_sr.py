import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from utility import *
from hsi_setup import Engine


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

scale_factors = [2]

def _parse_str_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(description='Hyperspectral Image Super-resolution')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--prefix', '-p', type=str, default='SR', help='name of task')
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='training batch size. Default=16')
    parser.add_argument('--nEpochs', '-n', type=int, default=50, help='number of epochs to train for. Default=50')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4.')
    parser.add_argument('--min-lr', '-mlr', type=float, default=5e-5, help='Minimal Learning Rate. Default=5e-5.')
    parser.add_argument('--ri', type=int, default=10, help='Record interval. Default=10')
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=2021')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)

    print(opt)

    cuda = opt.no_cuda

    """Setup Engine"""
    engine = Engine(opt.prefix, opt)
    scheduler = ReduceLROnPlateau(engine.optimizer, 'min', factor=0.1, patience=5, min_lr=opt.min_lr, verbose=True)

    use_2dconv = engine.net.module.use_2dconv if len(opt.gpu_ids) > 1 else engine.net.use_2dconv
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())
    
    sr_degrade = Compose([
        GaussianBlur(ksize=8, sigma=3),
        SequentialSelect([
            SRDegrade(sf) for sf in scale_factors
        ])
    ])

    common_transform = lambda x: x

    train_transform = Compose([
        sr_degrade,
        HSI2Tensor()
    ])

    valid_transform = Compose([
        sr_degrade,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31 = LMDBDataset('/media/liangzhiyuan/liangzy/qrnn3d/data/ICVL64_31_100.db')
    """Split patches dataset into training, validation parts"""
    icvl_64_31 = TransformDataset(icvl_64_31, common_transform)

    # Split the training and validating datasets.
    icvl_64_31_T, icvl_64_31_V = get_train_valid_dataset(icvl_64_31, 1000)

    train_dataset = ImageTransformDataset(icvl_64_31_T, train_transform)
    valid_dataset = ImageTransformDataset(icvl_64_31_V, valid_transform)

    icvl_64_31_TL = DataLoader(train_dataset,
                    batch_size=opt.batchSize, shuffle=True,
                    num_workers=opt.threads, pin_memory=cuda)

    icvl_64_31_VL = DataLoader(valid_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=opt.threads, pin_memory=cuda)


    adjust_learning_rate(engine.optimizer, opt.lr)
    while engine.epoch < opt.nEpochs:
        engine.train(icvl_64_31_TL)
        psnr, loss = engine.validate(icvl_64_31_VL)
            
        scheduler.step(loss)
        lrs = display_learning_rate(engine.optimizer)
        if engine.epoch % opt.ri == 0:
            engine.save_checkpoint(psnr, loss)

