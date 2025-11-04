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


out_dir = './model_path_EDRV/EDRV/'


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)

    DSC = 0.0
    print(num1)

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3 = model(image)
        # eval Dice
        res = F.upsample(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
        print(i, dice)
    return DSC / num1


def train(train_loader, test_loader, model, optimizer, epoch, test_path):
    model.train()
    global DSC_val_best
    loss_train = 0
    loss_segmentation = 0
    global best
    for step, (patch, mask) in enumerate(train_loader):
        patch = patch.cuda()
        mask = mask.cuda()
        # mask = torch.unsqueeze(mask,1)
        # print(patch.shape)
        # print(mask.shape)
        feature = net.forward(patch, False)
        # print(np.max(feature.detach().cpu().numpy()))
        # print(feature.shape) # 20, 32, 400, 400
        # print(mask.shape)
        # elbo = net.elbo(mask)
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
    loss_val = 0
    DSC_val = 0
    count = 0
    with torch.no_grad():
        for step, (patch, mask) in enumerate(test_loader):
            patch = patch.cuda()
            mask = mask.cuda()
            mask = torch.unsqueeze(mask, 1)
            output_samples = []
            for i in range(samples_per_example):
                output_samples.append(net.forward(patch, False))

            for k in range(patch.shape[0]):  # for all items in batch
                mask_out = mask[k, 0, :, :].detach().cpu().numpy()
                # pred_out = pred_mask[k, 0, :,:].detach().cpu().numpy()

                mask_out[mask_out >= 0.5] = 1
                mask_out[mask_out < 0.5] = 0
                result = []
                for j in range(len(output_samples)):  # for all output samples
                    pred = output_samples[j][k, 0, :, :]
                    pred = pred.sigmoid().data.cpu().numpy().squeeze()
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    # print(np.unique(pred))
                    result.append(dc(pred, mask_out))
                print(step, count, np.max(result))
                DSC_val += np.max(result)
                count = count + 1
    DSC_val /= count
    # train_loss.append(loss_train)
    print('End of epoch ', epoch + 1, ' , Train loss: ', loss_train, ', val DSC: ', DSC_val, 'best_DSC_before',
          DSC_val_best)
    secheduler.step()

    # save best model checkpoint
    if DSC_val > DSC_val_best:
        DSC_val_best = DSC_val
        fname = 'EDRV_model_dict_best_' + str(epoch) + '_' + str(DSC_val_best)[:5] + '.pth'
        torch.save(net.state_dict(), os.path.join(out_dir, fname))
        print('model saved at epoch: ', epoch + 1)
    else:
        fname = 'EDRV_model_dict_' + str(epoch) + '.pth'
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
                        default=5 * 1e-5, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

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

    parser.add_argument('--test_path', type=str,
                        default='../HSNet/DataSet_Segmentation/TestDataset-saEDRV/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth_saEDRV/' + model_name + '/')
    lr = 5 * 1e-5
    l2_reg = 5 * 1e-4
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
    best = 0

    # params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
        secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)

    image_root_test = '{}/images/'.format(opt.test_path)
    gt_root_test = '{}/masks/'.format(opt.test_path)

    test_loader = get_loader(image_root_test, gt_root_test, batchsize=1, trainsize=opt.trainsize,
                             augmentation=opt.augmentation)

    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    global DSC_val_best
    DSC_val_best = -999
    # for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    for epoch in range(1, opt.epoch):
        if epoch in [15, 30]:
            adjust_lr(optimizer, 0.5)
        train(train_loader, test_loader, net, optimizer, epoch, opt.test_path)
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
