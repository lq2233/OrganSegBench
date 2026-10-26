import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import HSNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient
import torch.nn.functional as F
import numpy as np
import logging
from medpy.metric.binary import dc

# checkpoint directory
out_dir = './model_pth_pancreas_for_test/PolypPVT_pancreas/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def structure_loss(pred, mask):
    # plain BCE + IoU (Eq. 2 in the paper), no boundary weighting
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred_s = torch.sigmoid(pred)
    inter = (pred_s * mask).sum(dim=(2, 3))
    union = (pred_s + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return bce + iou.mean()


def train(train_loader, val_loader, model, optimizer, epoch):
    model.train()
    # multi-scale training (PvT feature map sizes are multiples of 32)
    size_rates = [0.75, 1, 1.25]
    loss_epoch = 0.0
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images, gts = images.cuda(), gts.cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # M = 4 deep-supervision outputs
            P1, P2, P3, P4 = model(images)
            loss = structure_loss(P1, gts) + structure_loss(P2, gts) \
                   + structure_loss(P3, gts) + structure_loss(P4, gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
        loss_epoch += loss.item()
    loss_epoch /= len(train_loader)

    # validation: monitoring only, never used for model selection / early stopping
    model.eval()
    DSC_val, count = 0.0, 0
    with torch.no_grad():
        for images, gts in val_loader:
            images = images.cuda()
            P1, P2, P3, P4 = model(images)
            # average the four logits, then sigmoid: sigma(1/M * sum_m S_m)
            res = F.upsample((P1 + P2 + P3 + P4) / 4, size=(gts.shape[2], gts.shape[3]),
                             mode='bilinear', align_corners=False)
            res_np = (res.sigmoid().cpu().numpy()[0, 0] >= 0.5).astype(np.float32)
            gt_np = (gts[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
            DSC_val += dc(res_np, gt_np)
            count += 1
    if count > 0:
        DSC_val /= count
    print('End of epoch', epoch, ' loss:', loss_epoch, ' val DSC:', DSC_val)
    secheduler.step()

    if epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(out_dir, 'HSNet_model_dict_%d.pth' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./DataSet_Segmentation/TrainDataset-pancreas/',
                        help='path to training dataset (consensus pseudo-labels)')
    parser.add_argument('--val_path', type=str,
                        default='./DataSet_Segmentation/ValDataset-pancreas/',
                        help='path to validation dataset (manual GT, monitoring only)')
    opt = parser.parse_args()

    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build model ----
    model = HSNet().cuda()

    # ---- optimizer / scheduler (fixed schedule, no test-set based decay) ----
    lr = 1e-4
    l2_reg = 1e-5
    lr_decay_every = 200
    lr_decay = 0.95
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
    print(optimizer)

    # ---- data loaders (train / val are strictly separated) ----
    train_loader = get_loader(opt.train_path + 'images/', opt.train_path + 'masks/',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              shuffle=True, augmentation=False)
    val_loader = get_loader(opt.val_path + 'images/', opt.val_path + 'masks/',
                            batchsize=1, trainsize=opt.trainsize,
                            shuffle=False, augmentation=False)

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, val_loader, model, optimizer, epoch)

    # fixed-epoch final checkpoint (no best-model selection on the test set)
    torch.save(model.state_dict(), os.path.join(out_dir, 'HSNet_model_dict_1000_final.pth'))
