# train a probabilistic U-Net model

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_LIDC_data import LIDC_IDRI, LIDC_IDRI_train_New, LIDC_IDRI_test_New
from probabilistic_unet import ProbabilisticUnet
import pickle
import os
from medpy.metric.binary import dc, hd, hd95
# optimization settings
lr = 1e-5
l2_reg = 1e-6
lr_decay_every = 200   # decay LR after this many epochs
lr_decay = 0.95

batch_size_train = 5
batch_size_val = 1
epochs = 1000

# checkpoint directory
out_dir = './Model_path/AO/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('Folder already exists. Existing models and training logs will be replaced')
    
# data
dataset = LIDC_IDRI_train_New(dataset_location = './Data_Sample/TrainDataset/AO/')
dataset_test = LIDC_IDRI_test_New(dataset_location = './Data_Sample/TestDataset/AO/')

dataset_size = len(dataset)
dataset_size_test = len(dataset_test)

indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))

#np.random.shuffle(indices)
print('There is no random shuffle: initial portion of the dataset is used for train and the last portion for validation')

train_indices, test_indices = indices[:], indices[:split+1]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size_train, sampler=train_sampler)

test_loader = DataLoader(dataset_test, batch_size=batch_size_val, sampler=test_sampler)

print("Number of training/test patches:", (len(train_indices),len(test_indices)))

# network
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.cuda()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)

# logging
train_loss = []
test_loss = []
best_val_loss = 999.0
DSC_val_best = -999
for epoch in range(epochs):
    net.train()
    loss_train = 0
    loss_segmentation = 0
    # training loop
    for step, (patch, mask, _, _) in enumerate(train_loader): 
        patch = patch.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        loss = -elbo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_train += loss.detach().cpu().item()
        
        if step%10==0:
            print('[Ep ', epoch+1, (step+1), ' of ', len(train_loader) ,'] train loss: ', loss_train/(step+1))
        
    # end of training loop
    loss_train /= len(train_loader)
    samples_per_example = 10
    # valdiation loop
    net.eval()
    loss_val = 0
    DSC_val = 0
    count = 0
    with torch.no_grad():
        for step, (patch, mask, _, _) in enumerate(test_loader): 
            patch = patch.cuda()
            mask = mask.cuda()
            mask = torch.unsqueeze(mask,1)
            output_samples = []
            for i in range(samples_per_example): 
                net.forward(patch, mask, training=False)
                output_samples.append(torch.sigmoid(net.sample()).detach().cpu().numpy())
                
            for k in range(patch.shape[0]):    # for all items in batch
                patch_out = patch[k, 0, :, :].detach().cpu().numpy()
                mask_out = mask[k, 0, :, :].detach().cpu().numpy()

                mask_out[mask_out>=0.5]=1
                mask_out[mask_out<0.5]=0
                # if len(np.unique(mask_out))>1:
                result = []
                for j in range(len(output_samples)):  # for all output samples
                    pred = output_samples[j][k, 0, :, :]
                    pred[pred>=0.5]=1
                    pred[pred<0.5]=0
                    result.append(dc(pred, mask_out))
                print(step, count, np.max(result))
                DSC_val += np.max(result)
                count = count + 1

    DSC_val /= count
    train_loss.append(loss_train)
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val DSC: ', DSC_val, 'best_DSC_before', DSC_val_best)   
    
    secheduler.step()
    
    # save best model checkpoint
    if DSC_val > DSC_val_best:
        DSC_val_best = DSC_val
        fname = 'AO_model_dict_best_'+str(epoch)+'_'+str(DSC_val_best)[:6]+'.pth'
        # torch.save(net.state_dict(), os.path.join(out_dir, fname))
        print('model saved at epoch: ', epoch+1)
    else:
        fname = 'AO_model_dict_'+str(epoch)+'.pth'
        # torch.save(net.state_dict(), os.path.join(out_dir, fname))

print('Finished training')
# save loss curves        
plt.figure()
plt.plot(train_loss)
plt.title('train loss')
fname = os.path.join(out_dir, 'loss_train.png')
plt.savefig(fname)
plt.close()

# Saving logs
log_name = os.path.join(out_dir, "logging.txt")
with open(log_name, 'w') as result_file:
    result_file.write('Logging... \n')
    # result_file.write('Validation loss ')
    # result_file.write(str(test_loss))
    result_file.write('\nTraining loss  ')
    result_file.write(str(train_loss))