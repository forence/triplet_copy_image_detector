from __future__ import division, print_function
import os

import torch
import torch.nn as nn
import torch.optim as optims
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import TripletDatasets
from network import TripletNet
from trainer import fit

if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }

    # build up triplet data sets and data loaders

    data_dir = './copy_image_data'
    triplet_image_datasets = {x: TripletDatasets(os.path.join(data_dir, x), transform=data_transforms[x])
                              for x in ['train', 'val']}
    print(len(triplet_image_datasets['train']))
    print(len(triplet_image_datasets['val']))
    # # Test : figure anchor, positive, negative images
    # img = triplet_image_datasets['val'][np.random.randint(10)]
    # tsf = transforms.ToPILImage()
    # for i in range(3):
    #     tsf(img[i]).show()
    #     print(img[i].size())
    # # Test END

    use_gpu = torch.cuda.is_available()
    batch_size = 8
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    triplet_image_loaders = {x: DataLoader(triplet_image_datasets[x], batch_size=batch_size,
                                           shuffle=True if x == 'train' else False, **kwargs)
                             for x in ['train', 'val']}

    # set up the network and training parameters

    margin = 1.
    lr = 1e-3
    model = TripletNet()
    if use_gpu:
        model.cuda()

    loss_fn = nn.TripletMarginLoss(margin=margin)
    optimizer = optims.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
    num_epoch = 25
    log_interval = 5

    fit(triplet_image_loaders['train'], triplet_image_loaders['val'], model,
        loss_fn, optimizer, scheduler, num_epoch, use_gpu, log_interval)