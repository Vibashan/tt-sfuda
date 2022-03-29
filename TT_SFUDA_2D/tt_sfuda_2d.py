
import os
import yaml
import argparse

from glob import glob
from tqdm import tqdm
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from collections import OrderedDict

from albumentations import RandomRotate90,Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as st_transforms

import losses
import archs

cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None, help='model name')
    parser.add_argument('--target', default=None, help='model name')
    args = parser.parse_args()
    return args

def build_strong_augmentation(img):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    augmentation = []
    augmentation.append(st_transforms.RandomApply([st_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6))
    augmentation.append(st_transforms.RandomGrayscale(p=0.2))
    strong_aug = st_transforms.Compose(augmentation)
    s_input = strong_aug(img)
    return s_input

def build_pseduo_augmentation(img):
    aug1 = st_transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)
    aug2 = st_transforms.RandomGrayscale(p=1.0)
    aug3 = st_transforms.RandomSolarize(threshold=192.0,p=1.0)
    aug4 = st_transforms.RandomAutocontrast(p=1.0)

    aug_img1 = aug1(img).unsqueeze(0)
    aug_img2 = aug2(img).unsqueeze(0)
    aug_img3 = aug3(img).unsqueeze(0)
    aug_img4 = aug4(img).unsqueeze(0)
    aug_data = torch.cat([img.unsqueeze(0), aug_img1, aug_img2, aug_img3, aug_img4], dim=0)
    return aug_data

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

def consistency_loss(msrc_feat, tgt_feat):
    req_feat = [0,1,2,3]
    total_loss = 0 
    loss = nn.MSELoss()
    for i in req_feat:
        total_loss = total_loss + loss(tgt_feat[i], msrc_feat[i])
    return total_loss/len(req_feat)


@torch.jit.script
def sigmoid_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30)).mean()

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30))


def ent_select(aug_all_ent):
    aug_req_ent = []
    for i in range(len(aug_all_ent)): 
        if (aug_all_ent[i]).mean().item() > 0.0001: 
            aug_req_ent.append(aug_all_ent[i])
    return aug_req_ent

def uncert_voting(aug_output):
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = torch.sigmoid(aug_output[i])
        aug_all_prob.append(prob)
        aug_all_ent.append(sigmoid_entropy(prob))
    
    no_aug_prob_nor = torch.sigmoid(aug_output[0])
    no_aug_pseudo_label = no_aug_prob_nor.clone()
    no_aug_pseudo_label[no_aug_pseudo_label>=0.5]=1
    no_aug_pseudo_label[no_aug_pseudo_label<0.5]=0

    no_aug_ent = sigmoid_entropy(torch.sigmoid(aug_output[0]))
    no_aug_ent[torch.isnan(no_aug_ent)] = 0

    aug_prob_nor = sum(aug_all_prob)/len(aug_all_prob)
    aug_pseudo_label = aug_prob_nor
    aug_pseudo_label[aug_pseudo_label>=0.5]=1
    aug_pseudo_label[aug_pseudo_label<0.5]=0

    aug_req_ent = ent_select(aug_all_ent)
    aug_avg_ent = sum(aug_req_ent)/len(aug_req_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0

    no_aug_ent_nor = ((no_aug_ent - no_aug_ent.min()) * (1/(no_aug_ent.max() - no_aug_ent.min())))
    aug_avg_ent_nor = ((aug_avg_ent - aug_avg_ent.min()) * (1/(aug_avg_ent.max() - aug_avg_ent.min())))

    ent_weight = 0.75
    weighted_ent = ent_weight*no_aug_ent_nor+(1-ent_weight)*aug_avg_ent_nor
    weighted_ent_thresh = weighted_ent.clone()
    weighted_ent_thresh[weighted_ent_thresh>0.5]=1
    weighted_ent_thresh[weighted_ent_thresh<=0.5]=0

    prob_min = 0.3 
    unct_no_aug_prob = no_aug_prob_nor.clone()
    unct_no_aug_prob[unct_no_aug_prob>0.5]=0
    unct_no_aug_prob[unct_no_aug_prob<=prob_min]=0
    unct_no_aug_prob[unct_no_aug_prob>0]=1

    pseudo_uncert = unct_no_aug_prob.int()&weighted_ent_thresh.int()
    pseudo_label = no_aug_pseudo_label.int()|pseudo_uncert.int()
    return pseudo_label.unsqueeze(0).float()

def sfuda_target(config, train_loader, pseduo_model, msrc_model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    pseduo_model.eval()
    msrc_model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, path in train_loader:
        aug_input = build_pseduo_augmentation(input.squeeze(0))
        with torch.no_grad():
            aug_output = pseduo_model(aug_input.cuda())
            ps_output = uncert_voting(aug_output.detach())

        optimizer.zero_grad()
        output = msrc_model(aug_input.cuda())
        seg_loss = criterion(output.cuda(), ps_output.repeat(5,1,1,1).cuda())
        ent_loss = sigmoid_entropy_loss(torch.sigmoid(output))
        loss = seg_loss + ent_loss 
        loss.backward()
        optimizer.step()

        iou,dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def sfuda_task(train_loader, msrc_model, tgt_model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    msrc_model.eval()
    tgt_model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        w_input = input.cuda()
        target = target.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            w_output, msrc_feat = msrc_model(w_input, mode='const')
            ps_output = torch.sigmoid(w_output).detach().clone()
            ps_output[ps_output>=0.5]=1
            ps_output[ps_output<0.5]=0

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')
        seg_loss = criterion(output, ps_output)
        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss
        loss.backward()
        optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        new_msrc_dict = update_teacher_model(tgt_model, msrc_model, keep_rate=0.99)
        msrc_model.load_state_dict(new_msrc_dict)
        
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main():
    args = parse_args()

    config_file = "config_" + args.target
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', args.target, 'train','images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', args.target, 'test','images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', args.target, 'train','images'),
        mask_dir=os.path.join('inputs', args.target, 'train','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.target,'test', 'images'),
        mask_dir=os.path.join('inputs', args.target,'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    print("Creating model %s...!!!" % config['arch'])
    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    msrc_model.load_state_dict(torch.load('models/%s/model.pth'%config['name']))
    msrc_model.cuda()
    msrc_model.train()
    print("Sucessfully loaded source trained model...!!!")

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    tgt_model.cuda()
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=config['lr'], weight_decay=config['weight_decay'])

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=config['lr'], weight_decay=config['weight_decay'])

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.cuda()
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()
    
    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    print('Source_only dice: %.4f' % (val_log['dice']))

    print("")
    print("Target specific adaptation...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.cuda()
    tgt_model.train()

    print("")
    print("Task specific adaptation...!!!")
    for epoch in range(config['stage2']):
        train_log = sfuda_task(train_loader, msrc_model, tgt_model, criterion, tgt_optimizer)
        print('train_loss %.4f - train_iou %.4f'% (train_log['loss'], train_log['iou']))
    
    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model dice: %.4f' % (val_log['dice']))


if __name__ == '__main__':
    main()
