import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from dataloader import get_loader, test_dataset
import torch.nn.functional as F
import numpy as np
from unet import Unet
import logging
from medpy.metric.binary import dc, hd, hd95
import matplotlib.pyplot as plt


def adjust_lr(optimizer, decay_rate=0.1):
    # decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


out_dir = './model_path_EDRV_for_test/EDRV/'


def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred_s = torch.sigmoid(pred)
    inter = (pred_s * mask).sum(dim=(2, 3))
    union = (pred_s + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return bce + iou.mean()



def train(train_loader, test_loader, model, optimizer, epoch):
    model.train()

    loss_train = 0
    loss_segmentation = 0

    for step, (patch, mask) in enumerate(train_loader):
        patch = patch.cuda()
        mask = mask.cuda()

        feature = net.forward(patch, False)

        loss = structure_loss(feature, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.detach().cpu().item()

        if step % 100 == 0:
            print('[Ep ', epoch + 1, (step + 1), ' of ', len(train_loader), '] train loss: ', loss_train / (step + 1))

    loss_train /= len(train_loader)
    samples_per_example = 1
    # valdiation loop
    net.eval()
    DSC_val, count = 0, 0
    with torch.no_grad():
        for patch, mask in val_loader:
            patch, mask = patch.cuda(), mask.cuda()
            prob = torch.sigmoid(net.forward(patch, False))
            for k in range(patch.shape[0]):
                mask_out = (mask[k, 0].cpu().numpy() >= 0.5).astype(np.float32)
                pred = (prob[k, 0].cpu().numpy() >= 0.5).astype(np.float32)
                DSC_val += dc(pred, mask_out); count += 1
    DSC_val /= count
    print('End of epoch ', epoch, ' Train loss: ', loss_train, ' val DSC: ', DSC_val)
    secheduler.step()

    if epoch % 50 == 0:
        fname = 'EDRV_model_dict_%d.pth' % epoch
        torch.save(net.state_dict(), os.path.join(out_dir, fname))
    



if __name__ == '__main__':
    # dict_plot = {'saEDRV': []}
    # name = ['saEDRV']
    ##################model_name#############################
    model_name = 'EDRV_model_dict'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=1000, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='random flip/rotation')


    parser.add_argument('--batchsize', type=int,
                        default=20, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='../HSNet//DataSet_Segmentation/TrainDataset-saEDRV/',
                        help='path to train dataset')
    parser.add_argument('--val_path', type=str,
                    default='../HSNet/DataSet_Segmentation/ValDataset-saEDRV/')


    parser.add_argument('--train_save', type=str,
                        default='./model_pth_saEDRV/' + model_name + '/')
    lr = 1e-4
    l2_reg = 1e-5
    lr_decay_every = 200
    lr_decay = 0.95
    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    net = Unet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192],
               initializers={'w': 'he_normal', 'b': 'normal'}, apply_last_layer=True, padding=True)
    net.cuda()


    # params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2_reg)
        secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, weight_decay=1e-5, momentum=0.9)

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)

    image_root_val = '{}/images/'.format(opt.val_path)
    gt_root_val = '{}/masks/'.format(opt.val_path)
    val_loader = get_loader(image_root_val, gt_root_val, batchsize=1,
                            trainsize=opt.trainsize, shuffle=False, augmentation=False)

    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    # for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, val_loader, net, optimizer, epoch)
    torch.save(net.state_dict(), os.path.join(out_dir, 'EDRV_model_dict_1000_final.pth'))
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
