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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP


from augmentation.medical_augment import LmsDetectTrainTransform, LmsDetectTestTransform

import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar
import losses

state = {}
best_loss = 0
use_cuda = False


def main(config_file):
    global state, best_loss, use_cuda

    # initial distributed settings
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    # setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    if rank == 0:
        if not os.path.isdir(common_config['save_path']):
            mkdir_p(common_config['save_path'])
        # logger
        title = 'Chest landamrks detection using' + \
            common_config['arch']
        logger = Logger(os.path.join(
            common_config['save_path'], 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])

    augment_config = config['augmentation']
    # Dataset and Dataloader
    transform_train = LmsDetectTrainTransform(augment_config['rotate_angle'], augment_config['offset'])
    transform_test = LmsDetectTestTransform()
    data_config = config['dataset']
    print('==> Preparing dataset %s' % data_config['type'])
    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], transform_train,
        prefix=data_config['prefix'])
    testset = dataset.__dict__[data_config['type']](
        data_config['test_list'], data_config['test_meta'], transform_test,
        prefix=data_config['prefix'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset, shuffle=False)
    
    trainloader = dataset.DataLoaderX(local_rank=local_rank, dataset=trainset, batch_size=common_config['train_batch'],
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    testloader = dataset.DataLoaderX(local_rank=local_rank, dataset=testset, batch_size=common_config['test_batch'],
        sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=False)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](
        num_classes=data_config['num_classes'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    for ps in model.parameters():
        dist.broadcast(ps, 0)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank])

    # optimizer and scheduler
    state['lr'] = common_config['lr']
    criterion = losses.__dict__[config['loss_config']['type']]()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=0.9,
        weight_decay=common_config['weight_decay'])

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        train_loss = train(
            trainloader, model, criterion, optimizer, use_cuda, scaler)
        test_loss = test(testloader, model, criterion, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss])
        # save model
        is_best = test_loss < best_loss
        best_loss = max(test_loss, best_loss)
        
        if rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_path=common_config['save_path'])

    if rank == 0:
        logger.close()
        print('Best acc:' + str(best_loss))


def train(trainloader, model, criterion, optimizer, use_cuda, scaler):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    end        = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
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
        if int(os.environ['RANK']) == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.2f' % (losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def test(testloader, model, criterion, use_cuda):
    global best_acc
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


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
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()
    main(args.config_file)
