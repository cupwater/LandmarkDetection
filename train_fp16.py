'''
Training script for Landmark Detection of Medical Image
Copyright (c) Pengbo, 2022
'''
from __future__ import print_function

import os
import shutil
import time
import yaml

import torch
import torch.utils.data as data
import torch.optim as optim

from augmentation.medical_augment import LmsDetectTrainTransform, LmsDetectTestTransform
import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar, visualize_heatmap, get_landmarks_from_heatmap
import losses
import cv2

state = {}
best_loss = 0
use_cuda = True
exectime = time.time()

def main(config_file):
    global state, best_loss, use_cuda

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    if not os.path.isdir(common_config['save_path']):
        mkdir_p(common_config['save_path'])
    # logger
    title = 'Chest landamrks detection using' + \
        common_config['arch']
    logger = Logger(os.path.join(
        common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Avg-Train Loss', 'Avg-Valid Loss', 'Epoch-Train Loss', 'Epoch-Test Loss'])


    # initial dataset and dataloader
    augment_config = config['augmentation']
    # Dataset and Dataloader
    transform_train = LmsDetectTrainTransform(augment_config['rotate_angle'], augment_config['offset'])
    transform_test = LmsDetectTestTransform()
    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], augment_config,
        prefix=data_config['prefix'])
    testset = dataset.__dict__[data_config['type']](
        data_config['test_list'], data_config['test_meta'], {'rotate_angle': 0, 'offset': [0,0]},
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
    if use_cuda:
        model = model.cuda()

    # optimizer and scheduler
    state['lr'] = common_config['lr']
    criterion = losses.__dict__[config['loss_config']['type']]()
    
    optimizer = optim.Adam(
       filter(
           lambda p: p.requires_grad,
           model.parameters()),
        lr=common_config['lr'],
        weight_decay=common_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **common_config[common_config['scheduler_lr']])

    if args.visualize:
        checkpoints = torch.load(os.path.join(common_config['save_path'], 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoints['state_dict'], False)
        test(testloader, model, criterion, use_cuda, common_config, visualize=args.visualize)
        return

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True) if config['common']['fp16'] == True else None
    # Train and val
    for epoch in range(common_config['epoch']):
        print('\nEpoch: [%d | %d] LR: %f' %
                (epoch + 1, common_config['epoch'], state['lr']))
        train_loss, ep_train_loss = train(trainloader, model, criterion, optimizer, use_cuda, scaler, scheduler)
        test_loss, ep_test_loss = test(testloader, model, criterion, use_cuda, common_config, scaler, args.visualize)
        # save model
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, ep_train_loss, ep_test_loss])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])
        
        
    logger.close()
    print('Best loss:' + str(best_loss))


def train(trainloader, model, criterion, optimizer, use_cuda, scaler=None, scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    end        = time.time()

    for batch_idx, datas in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(datas) == 3:
            inputs, targets, masks = datas
            if use_cuda:
                masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
        else:
            inputs, targets= datas 
            masks = None
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        if scaler is None:
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks) / (outputs.size(0)*outputs.size(1))
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks) / (outputs.size(0)*outputs.size(1))
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f' % (losses.avg))
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, loss.item()


def test(testloader, model, criterion, use_cuda, common_config, scaler=None, visualize=None):
    global best_acc
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    index = 0
    for batch_idx, datas in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if len(datas) == 3:
            inputs, targets, masks = datas
            if use_cuda:
                masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
        else:
            inputs, targets= datas 
            masks = None
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        # compute output
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks) / (outputs.size(0)*outputs.size(1))
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets, datas) / (outputs.size(0)*outputs.size(1))

        if visualize:
            save_folder = os.path.join(common_config['save_path'], 'results/')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(inputs.size(0)):
                landmarks = get_landmarks_from_heatmap(outputs[i].detach())
                visualize_img = visualize_heatmap(inputs[i], landmarks)
                save_path = os.path.join(save_folder, str(batch_idx*inputs.size(0) + i)+'.jpg')
                cv2.imwrite(save_path, visualize_img)

        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(testloader), 'Loss: %.2f' % (losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, loss.item()


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
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args.config_file)


