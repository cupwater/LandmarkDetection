'''
Training script for Landmark Detection of Medical Image
Copyright (c) Pengbo, 2022
'''
from __future__ import print_function

import os
import shutil
import time
import pdb
import yaml
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import models
import dataset
from utils import Logger, AverageMeter, progress_bar, visualize_heatmap, get_landmarks_from_heatmap
import losses


state = {}
best_loss = 10000
use_cuda = False


def main(config_file):
    global state, best_loss, use_cuda

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']
    common_config['save_path'] = os.path.dirname(config_file)

    state['lr'] = common_config['lr']
    use_cuda = torch.cuda.is_available()

    augment_config = config['augmentation']

    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], augment_config,
        prefix=data_config['prefix'])
    testset = dataset.__dict__[data_config['type']](
        data_config['test_list'], data_config['test_meta'], {
            'rotate_angle': 0, 'offset': [0, 0]},
        prefix=data_config['prefix'])

    # create dataloader for training and testing
    trainloader = data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    testloader = data.DataLoader(
        testset, batch_size=common_config['test_batch'], shuffle=False, num_workers=2)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](
        num_classes=data_config['num_classes'])
    model = torch.nn.DataParallel(model)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    # optimizer and scheduler
    criterion = losses.__dict__[config['loss_config']['type']]()
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        weight_decay=common_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, **common_config[common_config['scheduler_lr']])
    # optimizer = optim.SGD(
    #    filter(
    #        lambda p: p.requires_grad,
    #        model.parameters()),
    #    lr=common_config['lr'],
    #    momentum=0.9,
    #    weight_decay=common_config['weight_decay'])

    if args.visualize:
        checkpoints = torch.load(os.path.join(common_config['save_path'], 'model_best.pth.tar'))
        model.load_state_dict(checkpoints['state_dict'], False)
        _, landmarks_array = validate(
            testloader, model, criterion, use_cuda, common_config, args.visualize)
        save_folder = os.path.join(common_config['save_path'], 'results/')
        save_path = os.path.join(save_folder, 'pred_landmarks.txt')
        np.savetxt(save_path, landmarks_array, fmt='%.3f')
        with open(save_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(landmarks_array.shape[0]) + '\n' + content)

        return

    # logger
    title = 'Chest landamrks detection using' + \
        common_config['arch']
    logger = Logger(os.path.join(
        common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])

    # Train and val
    for epoch in range(common_config['epoch']):
        #adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        train_loss = train(
            trainloader, model, criterion, optimizer, use_cuda, scheduler)
        test_loss, _ = validate(testloader, model, criterion,
                             use_cuda, common_config)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss])
        # save model
        is_best = test_loss < best_loss
        best_loss = max(test_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])

    logger.close()
    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, use_cuda, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, datas in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(datas) == 3:
            inputs, targets, masks = datas
            if use_cuda:
                masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
        else:
            inputs, targets = datas
            masks = None
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets, masks) / \
            (outputs.size(0)*outputs.size(1))
        losses.update(loss.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def validate(testloader, model, criterion, use_cuda, common_config, visualize=False):
    global best_acc
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    landmarks_list = []

    for batch_idx, datas in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(datas) == 3:
            inputs, targets, masks = datas
            if use_cuda:
                masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
        else:
            inputs, targets = datas
            masks = None
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets, masks) / \
            (outputs.size(0)*outputs.size(1))
        losses.update(loss.item(), inputs.size(0))

        if visualize:
            save_folder = os.path.join(common_config['save_path'], 'results/')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(inputs.size(0)):
                landmarks = get_landmarks_from_heatmap(outputs[i].detach())
                visualize_img = visualize_heatmap(inputs[i], landmarks)
                save_path = os.path.join(save_folder, str(
                    batch_idx*inputs.size(0) + i)+'.jpg')
                cv2.imwrite(save_path, visualize_img)
                landmarks_list.append(landmarks)

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if visualize:
        landmarks_array = np.array(landmarks_list).reshape(
            len(landmarks_list), -1)
        return losses.avg, landmarks_array
    else:
        return losses.avg, None


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            save_path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, config):
    global state
    if epoch in config['scheduler']:
        state['lr'] *= config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Landmark Detection for Medical Image')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str,
                        default='experiments/template/landmark_detection_template.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--gpu-id', type=str, default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)
