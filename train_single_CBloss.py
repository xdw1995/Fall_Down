from TSM import TSN
import torch

import os
import time
import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_dataloader, epoch, criterion, optimizer):
    global niubi
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):

        data_time.update(time.time() - end)
        # print(inputs.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)

        samples_per_cls = torch.Tensor([sum(labels==i) for i in range(5)]).cuda()

        loss_type = "softmax"
        loss = criterion(labels, outputs, samples_per_cls, 5,loss_type)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step+1) % 2 == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
                break
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)




def validation(model, val_dataloader, epoch, criterion, optimizer):
    global niubi
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            samples_per_cls = torch.Tensor([sum(labels==i) for i in range(5)]).cuda()

            loss_type = "softmax"

            loss = criterion(labels, outputs, samples_per_cls, 5,loss_type)
            # print(loss)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            print('----validation----')
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
            # if int(top1.avg)>=niubi:
            #     niubi = int(top1.avg)
            #     torch.save(model.module.state_dict(), 'jiaochashang'+".pth.tar")



def main():
    from dataset import VideoDataset_single,VideoDataset_3_frames
    import warnings
    warnings.filterwarnings("ignore")
    from TSM import diff

    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # logdir = os.path.join(params['log'], cur_time)
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)


    print("Loading dataset")
    train_dataloader = \
        DataLoader(
            VideoDataset_single(mode='train'),
            batch_size=128, shuffle=True)

    val_dataloader = \
        DataLoader(
            VideoDataset_single(mode='test'),
            batch_size=64, shuffle=False)
    # test_dataloader = \
    #     DataLoader(
    #         VideoDataset(1,mode='test', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
    #         batch_size=32, shuffle=False, num_workers=params['num_workers'])

    model = TSN(5,1,base_model = 'resnet34',modality = 'RGB',partial_bn=True,is_shift=False)
    # print(model)
    # (layer3): Sequential(
    #     (0): Bottleneck(
    #     (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    politics =model.get_optim_policies()
    # #--gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    #
    pretrained_dict = torch.load('/data/xudw/temporal-shift-module-master/Fall_down_Single.pth.tar', map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("load pretrain model")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model = nn.DataParallel(model)  # multi-Gpu

    from loss_balance import CB_loss
    # criterion = nn.CrossEntropyLoss().cuda()
    beta = 0.9999
    gamma = 2.0
    criterion = CB_loss(beta,gamma).cuda()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(politics, lr=1e-2, momentum=0.9, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.1)

    for epoch in range(1000):
        train(model, train_dataloader, epoch, criterion, optimizer)
        # if epoch%2==0:
        validation(model, val_dataloader, epoch, criterion, optimizer)
        # if epoch % 2== 0:
        scheduler.step()
        # if epoch % 1 == 0:
        #     checkpoint = os.path.join(model_save_dir,
        #                               "clip_len_biaozhun" + str(params['clip_len']) + "frame_sample_rate_" +str(params['frame_sample_rate'])+ "_checkpoint_" + str(epoch) + ".pth.tar")
        #     torch.save(model.module.state_dict(), checkpoint)

if __name__ == '__main__':
    global niubi
    niubi =70
    main()
