import torch
import argparse
import os
import numpy as np
import random
from recognition.yolo_v1.yolo import YOLOV1
from tqdm import tqdm
from recognition.yolo_v1.dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from recognition.yolo_v1.loss import YOLOV1Loss
from recognition.yolo_v1.config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def collate_function(data):
    return list(zip(*data))


def train(config):
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'])
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                        num_classes=dataset_config['num_classes'],
                        model_config=model_config)
    yolo_model.train()
    yolo_model.to(device)
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ckpt_name'])):
        print('Loading checkpoint as one exists')
        yolo_model.load_state_dict(
            torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']), map_location=device))
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              yolo_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    criterion = YOLOV1Loss()
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    for epoch_idx in range(num_epochs):
        losses = []
        optimizer.zero_grad()
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            yolo_targets = torch.cat([
                target['yolo_targets'].unsqueeze(0).float().to(device)
                for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in ims], dim=0)
            yolo_preds = yolo_model(im)
            loss = criterion(yolo_preds, yolo_targets, use_sigmoid=model_config['use_sigmoid'])
            loss = loss / acc_steps
            loss.backward()
            losses.append(loss.item())
            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                print('Loss : {:.4f}'.format(np.mean(losses)))
            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)
            steps += 1
        print('Finished epoch {}'.format(epoch_idx + 1))
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.save(yolo_model.state_dict(), os.path.join(train_config['task_name'], train_config['ckpt_name']))
    print('Done Training...')


if __name__ == '__main__':
    train(config)
