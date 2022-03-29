
import os
import time
import logging
import argparse
import numpy as np

import models
import torchio as tio
from tqdm import tqdm
from utils import Parser
from data import datasets
from models import criterions
from data.data_utils import add_mask
from collections import OrderedDict

import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from data.sampler import CycleSampler
from data.data_utils import add_mask, init_fn

path = os.path.dirname(__file__)
cudnn.benchmark = True

eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0
    ret += f1_score(o, t),
    # core
    o = (output==1) | (output==4)
    t = (target==1) | (target==4)
    ret += f1_score(o , t),
    # active
    o = (output==4); t = (target==4)
    ret += f1_score(o , t),
    return ret


@torch.jit.script
def softmax_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).mean()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(0) * x.log_softmax(0))

def build_strong_augmentation(img):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    aug1 = tio.transforms.RandomMotion()
    aug2 = tio.transforms.RandomGhosting()
    aug3 = tio.transforms.RandomBiasField()
    aug4 = tio.transforms.RandomBlur()
    
    s_input = torch.tensor(aug4(aug3(aug2(aug1(img)))))
    return s_input.unsqueeze(1)

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    student_model_dict = model_student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    return new_teacher_dict

def build_pseduo_augmentation(img):
    aug1 = tio.transforms.RandomMotion()
    aug2 = tio.transforms.RandomGhosting()
    aug3 = tio.transforms.RandomBiasField()
    aug4 = tio.transforms.RandomBlur()

    aug_img1 = torch.tensor(aug1(img))
    aug_img2 = torch.tensor(aug2(img))
    aug_img3 = torch.tensor(aug3(img))
    aug_img4 = torch.tensor(aug4(img))
    aug_data = torch.cat([torch.tensor(img), aug_img1, aug_img2, aug_img3, aug_img4], dim=0)
    return aug_data.unsqueeze(1)

def consistency_loss(msrc_feat, tgt_feat):
    req_feat = [0,1]
    total_loss = 0 
    loss = nn.MSELoss()
    for i in req_feat:
        total_loss = total_loss + loss(tgt_feat[i], msrc_feat[i])
    return total_loss/len(req_feat)

def uncert_voting(aug_output):
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = F.softmax(aug_output[i], dim=0)
        aug_all_prob.append(prob)
        aug_all_ent.append(softmax_entropy(aug_output[i]))

    no_aug_prob_nor = F.softmax(aug_output[0], dim=0)
    no_aug_pseudo_label = no_aug_prob_nor.clone().detach().cpu().numpy()
    no_aug_pseudo_label = no_aug_pseudo_label.argmax(0) 

    no_aug_ent = softmax_entropy(aug_output[0])
    no_aug_ent[torch.isnan(no_aug_ent)] = 0

    aug_prob_nor = sum(aug_all_prob)/len(aug_all_prob)
    aug_pseudo_label = aug_prob_nor.clone().detach().cpu().numpy()
    aug_pseudo_label = aug_pseudo_label.argmax(0) 

    aug_avg_ent = sum(aug_all_ent)/len(aug_all_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0

    no_aug_ent_nor = ((no_aug_ent - no_aug_ent.min()) * (1/(no_aug_ent.max() - no_aug_ent.min())))
    aug_avg_ent_nor = ((aug_avg_ent - aug_avg_ent.min()) * (1/(aug_avg_ent.max() - aug_avg_ent.min())))

    ent_weight = 0.75
    weighted_ent = ent_weight*no_aug_ent_nor+(1-ent_weight)*aug_avg_ent_nor
    weighted_ent_thresh = weighted_ent.clone()
    weighted_ent_thresh[weighted_ent_thresh>0.5]=1
    weighted_ent_thresh[weighted_ent_thresh<=0.5]=0

    prob_min = 0.3
    unct_no_aug_trans = no_aug_prob_nor.clone()
    unct_no_aug_trans[unct_no_aug_trans>0.5]=0
    unct_no_aug_trans[unct_no_aug_trans<=prob_min]=0
    unct_no_aug_trans[unct_no_aug_trans>0]=1
    unct_no_aug_prob = unct_no_aug_trans.argmax(0)

    unct_no_aug_prob = torch.tensor(unct_no_aug_prob).clone().detach().cuda()
    weighted_ent_thresh = torch.tensor(weighted_ent_thresh).clone().detach().cuda()
    no_aug_pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().cuda()

    pseudo_uncert = unct_no_aug_prob.int()&weighted_ent_thresh.int()
    pseudo_label = no_aug_pseudo_label.int()|pseudo_uncert.int()
    pseudo_label = pseudo_label.argmax(0)
    return pseudo_label.unsqueeze(0).long()


def sfuda_target(train_loader, pseduo_model, msrc_model, criterion, optimizer, break_pt, tgt_num):
    avg_meters = {'loss': AverageMeter()}
    pseduo_model.eval()
    msrc_model.cuda()
    msrc_model.train()
    pbar = tqdm(total=break_pt)
    for i, data in enumerate(train_loader):
        data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]
        input = x[:,tgt_num,:,:,:]
        aug_input = build_pseduo_augmentation(input.cpu().numpy())

        with torch.no_grad():
            aug_output = pseduo_model(aug_input.cuda())
            ps_output = uncert_voting(aug_output.detach())

        optimizer.zero_grad()
        output = msrc_model(aug_input.cuda())

        seg_loss = criterion(output.cuda(), ps_output.repeat(5,1,1,1).cuda())
        ent_loss = softmax_entropy_loss(output)
        loss = 0.01*seg_loss + ent_loss 
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
        if i == break_pt:
            break
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg)])

def sfuda_task(train_loader, msrc_model, tgt_model, criterion, optimizer, break_pt, tgt_num):
    avg_meters = {'loss': AverageMeter()}
    msrc_model.eval()
    tgt_model.train()
    pbar = tqdm(total=break_pt)
    for i, data in enumerate(train_loader):
        data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]
        w_input = x[:,tgt_num,:,:,:]
        image_strong_aug = build_strong_augmentation(w_input.cpu().numpy())
        s_input = image_strong_aug.cuda()

        with torch.no_grad():
            w_output, msrc_feat = msrc_model(w_input.unsqueeze(0).cuda(), mode='const')
            ps_output = F.softmax(w_output, dim=1)
            ps_output = ps_output[0,:,:,:,:].detach().cpu().numpy()
            ps_output = ps_output.argmax(0)
            ps_output = torch.tensor(ps_output).unsqueeze(0).cuda()

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')
        seg_loss = criterion(output, ps_output)
        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), w_input.size(0))
        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)

        new_msrc_dict = update_teacher_model(tgt_model, msrc_model, keep_rate=0.9)
        msrc_model.load_state_dict(new_msrc_dict)
        if i == break_pt:
            break 
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg)])

def validate(valid_loader, model, out_dir='', names=None, tgt_num=None, scoring=True, verbose=True, mode=None):
    H, W, T = 240, 240, 155
    model.eval()
    vals = AverageMeter()

    for k in range(tgt_num,tgt_num+1):
        for i, data in enumerate(valid_loader):
            target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
            if len(data) > 2:
                x = add_mask(x, data.pop(), 1)
            x =  x[:,k,:,:,:]
            x = x.unsqueeze(1)
            logit = model(x) 
            output = F.softmax(logit, dim=1) 
            output = output[0, :, :H, :W, :T].detach().cpu().numpy()
            msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
            if scoring:
                output = output.argmax(0)
                scores = dice(output, target_cpu)
                vals.update(np.array(scores))
                msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if scoring:
            if k==0:
                nam = "FLAIR"
            elif k==1:
                nam = "T1CE"
            elif k==2:
                nam  ="T1"
            elif k==3:
                nam = "T2"
            msg = 'Adapt to {}  '.format(nam)
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
            logging.info(msg)
            vals = AverageMeter()
    return vals.avg

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


keys = 'whole', 'core', 'enhancing', 'loss'
def main():
    ckpts = args.getdir()
    model_file = os.path.join(ckpts, args.ckpt)

    if args.cfg == 'unet_t1':
        tgt_num = 1
        print("Model trained on T1 is adapted to T1CE")
    elif args.cfg == 'unet_t2':
        tgt_num = 0
        print("Model trained on T2 is adapted to FLAIR")
    elif args.cfg == 'unet_flair':
        tgt_num = 3
        print("Model trained on FLAIR is adapted to T2")

    print("Loading source trained model...!!!")
    msrc_model = models.unet.Unet_wo()
    msrc_model.load_state_dict(torch.load(model_file)['state_dict'])
    msrc_model = msrc_model.cuda()
    msrc_model.train()
    print("Sucessfully loaded source trained model...!!!")

    tgt_model = models.unet.Unet_wo()
    
    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=args.opt_params.lr, weight_decay=0.0001)

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=args.opt_params.lr, weight_decay=0.0001)

    pseudo_model = models.unet.Unet_wo()
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.cuda()
    pseudo_model.eval()

    Dataset = getattr(datasets, args.dataset)
    train_list = os.path.join(args.data_dir, args.train_list)
    valid_list = os.path.join(args.data_dir, args.valid_list)
    train_set = Dataset(train_list, root=args.data_dir, for_train=True, transforms=args.train_transforms)
    break_pt=25

    num_iters = args.num_iters or (len(train_set) * 1) // args.batch_size
    num_iters -= args.start_iter

    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
    train_loader = DataLoader(
                    train_set,
                    batch_size=1,
                    collate_fn=train_set.collate, sampler=train_sampler,
                    num_workers=args.workers, pin_memory=True, worker_init_fn=init_fn)

    valid_set = Dataset(valid_list, root=args.data_dir,
                    for_train=False, return_target=args.scoring,
                    transforms=args.test_transforms)
    valid_loader = DataLoader(
                    valid_set,
                    batch_size=1, shuffle=False,
                    collate_fn=valid_set.collate,
                    num_workers=4, pin_memory=True)

    criterion = getattr(criterions, args.criterion)

    print("")
    print("Performing source only model evaluation...!!!")
    validate(valid_loader, pseudo_model, args.out_dir, valid_set.names, tgt_num=tgt_num, scoring=args.scoring, mode="tgt")
    
    print("")
    print("Target specific adaptation...!!!")
    start = time.time()
    for epoch in range(args.stage1):
        train_log = sfuda_target(train_loader, pseudo_model, msrc_model, criterion, src_optimizer, break_pt, tgt_num)
        print('loss %.4f - '% (train_log['loss']))
        torch.cuda.empty_cache()

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.cuda()
    tgt_model.train()

    print("")
    print("Task specific adaptation...!!!")
    for epoch in range(args.stage2):
        train_log = sfuda_task(train_loader, msrc_model, tgt_model, criterion, tgt_optimizer, break_pt, tgt_num)
        print('loss %.4f - '% (train_log['loss']))
        torch.cuda.empty_cache()

    print("")
    print("Performing adapted target model evaluation...!!!")
    validate(valid_loader, msrc_model, args.out_dir, valid_set.names, tgt_num=tgt_num, scoring=args.scoring, mode="ent")
    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', default='unet', type=str)
    args = parser.parse_args()
    args = Parser(args.cfg, log='test').add_args(args)

    #### Update the path correspondingly
    args.data_dir = '/media/vibsss/src_free/medical_src_free/github/TT_SFUDA_3D/data/train/all'
    args.valid_list = 'val.txt'
    args.saving = True
    args.scoring = True 
    args.ckpt = 'source_model.tar'

    if args.saving:
        folder = os.path.splitext(args.valid_list)[0]
        out_dir = os.path.join('output', args.name, folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
    else:
        args.out_dir = ''

    main()
