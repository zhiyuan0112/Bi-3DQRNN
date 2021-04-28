import os
import argparse

import torch
import torch.nn as nn

import models
from utility import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

scale_factors = [2]
prefix = 'SR'

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
    parser = argparse.ArgumentParser(description='Hyperspectral Image Denoising (Non-i.i.d.)')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2020, help='random seed to use. Default=2020')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')
    parser.add_argument('--test', action='store_true', help='test mode?')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--use-2dconv', action="store_true", help='whether the network uses 2d convolution?')
    parser.add_argument('--bandwise', action="store_true", help='whether the network handles the input in a band-wise manner?')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    print(opt)

    cuda = not opt.no_cuda
    use_2dconv = opt.use_2dconv

    """Prepare Data"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())

    common_transform = lambda x: x

    print('==> Preparing data..')

    mat_transforms = [
        Compose([
            common_transform,
            GaussianBlur(ksize=8, sigma=3),
            SRDegrade(sf),
            HSI2Tensor()
        ])
        for sf in scale_factors
    ]

    datadir = '/media/liangzhiyuan/liangzy/qrnn3d/data/icvl512_101_gt'
    mat_dataset = MatDataFromFolder(datadir, size=None)
    fns = os.listdir(datadir)
    mat_dataset.filenames = [os.path.join(datadir, fn) for fn in fns]
    mat_dataset = TransformDataset(mat_dataset, LoadMatKey(key='gt'))
    mat_datasets = [ImageTransformDataset(mat_dataset, mat_transform) for mat_transform in mat_transforms]
    mat_loaders = [DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=cuda
                ) for mat_dataset in mat_datasets]


    """Model"""
    print("==> Creating model '{}'..".format(opt.arch))
    net = models.__dict__[opt.arch]()
    criterion = nn.MSELoss()

    if len(opt.gpu_ids) > 1:
        from models.sync_batchnorm import DataParallelWithCallback
        net = DataParallelWithCallback(net, device_ids=opt.gpu_ids)

    if cuda:
        net.cuda()
        criterion.cuda()

    """Resume previous model"""
    print('==> Resuming from checkpoint %s..' %opt.resumePath)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.resumePath or './checkpoint/%s/%s/model_best.pth'%(opt.arch, prefix))
    net.load_state_dict(checkpoint['net'])


    """Testing"""
    def test(test_loader, scale):
        net.eval()
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        loss_data = 0
        cnt = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if not opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()                
            with torch.no_grad():
                bw_flag = opt.bandwise
                if bw_flag:
                    O = []
                    for time, (i, t) in enumerate(zip(inputs.split(1,1), targets.split(1,1))):
                        o = net(i)
                        O.append(o)
                        loss = criterion(o, t)
                        loss_data += loss.data[0]
                    outputs = torch.cat(O, dim=1)
                else:
                    outputs = net(inputs)
                    loss_data = criterion(outputs, targets)

            outputs = outputs.cpu()
            targets = targets.cpu()

            psnr,ssim,sam = MSIQA(outputs, targets)
            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)
            total_ssim += ssim
            avg_ssim = total_ssim / (batch_idx + 1)
            total_sam += sam
            avg_sam = total_sam / (batch_idx + 1)

            progress_bar(batch_idx, len(test_loader), 'PSNR: %.4f | SSIM: %.4f | SAM: %.4f '
                % (avg_psnr, avg_ssim, avg_sam))


            # Save .mat result.
            def torch2numpy(hsi):
                if use_2dconv:
                    R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
                else:
                    R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
                return R_hsi

            from os.path import join
            from scipy.io import savemat
            from os.path import basename, exists
            # filedir = '/data1/liangzhiyuan/Data/result'
            filedir = False
            if filedir:
                outpath = join(filedir, 'bi-3dqrnn-%d-%d.mat' % (cnt, scale))
                cnt += 1
                if not exists(filedir):
                    os.mkdir(filedir)
                if not exists(outpath):
                    savemat(outpath, {'pred': torch2numpy(targets)})


    for i, tl in enumerate(mat_loaders):
        test(tl, scale_factors[i])
