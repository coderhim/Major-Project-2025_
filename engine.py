from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice

from monai.losses import DiceLoss
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
from globals import train_dice_losses, train_cons_losses, train_lrs
print = functools.partial(print, flush=True)

dice_loss = DiceLoss(to_onehot_y=False,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)

class SemanticConsistencyLoss(nn.Module):
    def __init__(self, feat_layers=[1,3,5], weight=0.3):
        super().__init__()
        self.feat_layers = feat_layers
        self.weight = weight
        
    def forward(self, feats_orig, feats_aug):
        loss = 0
        for l in self.feat_layers:
            orig = feats_orig[l].flatten(2)
            aug = feats_aug[l].flatten(2)
            loss += F.mse_loss(orig, aug, reduction='none').mean([1,2])
        return self.weight * loss.mean()

def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()
    aux_criterion = SemanticConsistencyLoss()
    # aug_module = HybridAugmentor(num_classes=5)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 50
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
             # Generate augmented sample
            # augmented = aug_module(img, lbl, cur_iteration, warmup_iteration)
            # Forward passes
            logits_orig, feats_orig = model(img, return_features=True)
            # logits_aug, feats_aug = model(augmented, return_features=True)
            # Loss calculation
            dice_loss = criterion.get_loss(logits_orig, lbl) + criterion.get_loss(logits_aug, lbl)
            cons_loss = aux_criterion(feats_orig, feats_aug)

            total_loss = dice_loss + cons_loss

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update metrics
            metric_logger.update(dice_loss=dice_loss.item(), cons_loss=cons_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if cur_iteration >= warmup_iteration:
                print(f'WarmUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration

        metric_logger.synchronize_between_processes()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1, grad_scaler=None):
    model.train()
    criterion.train()
    aux_criterion = SemanticConsistencyLoss()
    # aug_module = HybridAugmentor(num_classes=5)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move samples to the appropriate device
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']

        # Generate augmented sample
        # augmented = aug_module(img, lbl, cur_iteration, max_iteration)

        if grad_scaler is None:
            # Regular precision
            logits_orig, feats_orig = model(img, return_features=True)
            logits_aug, feats_aug = model(augmented, return_features=True)

            dice_loss = criterion.get_loss(logits_orig, lbl) + criterion.get_loss(logits_aug, lbl)
            cons_loss = aux_criterion(feats_orig, feats_aug)
            total_loss = dice_loss + cons_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            # Mixed precision (AMP)
            with torch.cuda.amp.autocast():
                logits_orig, feats_orig = model(img, return_features=True)
                logits_aug, feats_aug = model(augmented, return_features=True)

                dice_loss = criterion.get_loss(logits_orig, lbl) + criterion.get_loss(logits_aug, lbl)
                cons_loss = aux_criterion(feats_orig, feats_aug)
                total_loss = dice_loss + cons_loss

            optimizer.zero_grad()
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        # Update metrics
        metric_logger.update(dice_loss=dice_loss.item(), cons_loss=cons_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        cur_iteration += 1
        if cur_iteration >= max_iteration and max_iteration > 0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return cur_iteration

def adaptive_threshold( epoch, total_epochs, tau_max=0.7, tau_min=0.3, gamma=5.0):
        """Sigmoidal curriculum learning for saliency threshold"""
        t = epoch / total_epochs
        return tau_min + (tau_max - tau_min) * (2/(1 + np.exp(-gamma*t)) - 1)

import torch
import numpy as np
import random

def class_aware_mixup_segmentation(images, masks, num_classes=5, alpha=0.4):
    """
    Applies Class-Aware Mixup for segmentation tasks with one-hot masks.

    - images: Tensor of shape (B, C, H, W) - batch of images
    - masks: Tensor of shape (B, num_classes, H, W) - one-hot encoded masks
    - num_classes: Total number of classes including background (default = 5)
    - alpha: Mixup Beta distribution parameter

    Returns:
    - Mixed images and mixed one-hot masks
    """

    batch_size = images.size(0)
    mixed_images = images.clone()
    mixed_masks = masks.clone()

    # Convert one-hot to class indices for easier class-aware selection
    mask_indices = torch.argmax(masks, dim=1)  # Shape: (B, H, W)

    for class_label in range(1, num_classes):  # Ignore class 0 (background)
        # Find samples containing this class
        class_indices = (mask_indices == class_label).any(dim=(1, 2)).nonzero(as_tuple=True)[0]
        if len(class_indices) < 2:  # Need at least two samples for mixup
            continue

        for idx in class_indices:
            mix_idx = random.choice(class_indices)
            if idx == mix_idx:  # Avoid self-mixing
                continue
            
            lam = np.random.beta(alpha, alpha)  # Mixup ratio

            # Mix images
            mixed_images[idx] = lam * images[idx] + (1 - lam) * images[mix_idx]
            
            # Mix labels using soft-labeling
            mixed_masks[idx] = lam * masks[idx] + (1 - lam) * masks[mix_idx]

    return mixed_images, mixed_masks

def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_epoch: int, max_iteration: int = -1,config=None,visdir=None):
    
    global train_lrs, train_cons_losses, train_dice_losses

    model.train()
    criterion.train()
    aux_criterion = SemanticConsistencyLoss()
    # aug_module = HybridAugmentor(num_classes=5).to(device)  # Move to the same device

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 10
    visual_freq = 100  # Added visual frequency
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move samples to the appropriate device
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
          # Apply Class-Aware Mixup
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            # visual_dict['GLA']=mixed_GLA_img.detach().cpu().numpy()[0,0]
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
            # visual_dict['classmixed_GT']=mixed_lbl.detach().cpu().numpy()[0]
        else:
            visual_dict=None

        # Generate augmented sample
        lbl = F.one_hot(lbl,5).permute((0,3,1,2))
        mixed_GLA_img, mixed_lbl = class_aware_mixup_segmentation(GLA_img, lbl, num_classes=5)
        print("here i am ",mixed_lbl.shape)
        input_var = Variable(mixed_GLA_img, requires_grad=True)
        if visual_dict is not None:
            visual_dict['GLA']=mixed_GLA_img.detach().cpu().numpy()[0,0]  
        # print("#####@here i sthe device ", device)
        # print(input_var.shape)
        # print(lbl.shape)
        # global_aug, local_aug = aug_module(img, lbl, cur_iteration, max_iteration)
        grad_scaler = None
        if grad_scaler is None:
            optimizer.zero_grad()
            logits_orig, feats_orig_list = model(input_var, return_features=True)
            # print(logits_orig.shape)
            losses=dice_loss(logits_orig, mixed_lbl)
            losses.backward(retain_graph=True)
            # print("LLA image size ", LLA_img.shape)

            # saliency
            gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()

            saliency=get_SBF_map(gradient,config.grid_size)

            if visual_dict is not None:
                visual_dict['GLA_pred']=torch.argmax(logits_orig,1).cpu().numpy()[0]

            # threshold_value = adaptive_threshold(epoch, max_epoch)
            # saliency = (saliency > threshold_value).float()
            if visual_dict is not None:
                visual_dict['GLA_saliency']= saliency.detach().cpu().numpy()[0,0]
            mixed_img = mixed_GLA_img.detach() * saliency + LLA_img * (1 - saliency)
            sum_lbl = mixed_lbl + lbl
            if visual_dict is not None:
                visual_dict['SBF']= mixed_img.detach().cpu().numpy()[0,0]

            aug_var = Variable(mixed_img, requires_grad=True)
            logits_aug, feats_aug_list = model(aug_var, return_features=True)
            dice_loss_value = dice_loss(logits_aug, sum_lbl)
            cons_loss = aux_criterion(feats_orig_list, feats_aug_list)
            total_loss = dice_loss_value + cons_loss
            total_loss.backward()
            # optimizer.step()

            if visual_dict is not None:
                visual_dict['SBF_pred'] = torch.argmax(logits_aug, 1).cpu().numpy()[0]

            optimizer.step()

            # Regular precision
            # logits_orig, feats_orig = model(img, return_features=True)
            # logits_aug, feats_aug = model(augmented, return_features=True)
            # logits_orig = model(img)
            # feats_orig_list = model.encoder(img)
            # feats_orig = feats_orig_list[-1] #last feature map
            # logits_aug = model(augmented)
            # feats_aug_list = model.encoder(augmented)
            # feats_aug = feats_aug_list[-1]
            # Single forward pass for original image
            # optimizer.zero_grad()
            # logits_orig, feats_orig_list = model(input_var, return_features=True)
            # losses=dice_loss(logits_orig, lbl)
            # logits = model(input_var)
            # loss_dict = criterion.get_loss(logits_orig, lbl)
            # losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            # losses.backward(retain_graph=True)

            # # saliency
            # gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()
            # saliency=get_SBF_map(gradient,config.grid_size)
            # if visual_dict is not None:
            #     visual_dict['Input_Img_saliency']= saliency.detach().cpu().numpy()[0,0]
            #     visual_dict['Input_Img_pred']=torch.argmax(logits_orig,1).cpu().numpy()[0]
            # # Single forward pass for augmented image
            # threshold_value = aug_module.adaptive_threshold(epoch, max_epoch)
            # saliency = (saliency > threshold_value).float()
            # mixed_img = global_aug * saliency + local_aug * (1 - saliency)
            # if visual_dict is not None:
            #     visual_dict['Augmented']= mixed_img.detach().cpu().numpy()[0,0]
            # augmented = Variable(mixed_img, requires_grad=True)
            # logits_aug, feats_aug_list = model(augmented, return_features=True)
            # # Note the change here - using the global dice_loss function but storing result in dice_loss_value
            # # print("model Output Shape : ", logits_orig.shape)
            # # print("model Output augmented Shape : ", logits_aug.shape)
            # # print("label Shape : ", lbl.shape)
            # # print("features Shape : ", feats_orig.shape)
            # # print("features augmented Shape : ", feats_aug.shape)
            # dice_loss_value = dice_loss(logits_aug, lbl) 
            # cons_loss = aux_criterion(feats_orig_list, feats_aug_list)
            # total_loss = dice_loss_value + cons_loss

            
            # total_loss.backward()
            # optimizer.step()
        else:
            # Mixed precision (AMP)
            # Ensure GradScaler is initialized before training starts
            # if "grad_scaler" not in globals():
            grad_scaler = torch.amp.GradScaler()

            # Mixed precision (AMP) training
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Single forward pass for original image
                logits_orig, feats_orig_list = model(GLA_img, return_features=True)
                # Single forward pass for augmented image
                logits_aug, feats_aug_list = model(GLA_img, return_features=True)

                # Compute losses
                dice_loss_value = (dice_loss(logits_orig, lbl) + dice_loss(logits_aug, lbl)) / 2
                cons_loss = aux_criterion(feats_orig_list, feats_aug_list)
                total_loss = dice_loss_value + cons_loss

            # Zero gradients
            optimizer.zero_grad()

            # Scale the loss and backpropagate
            grad_scaler.scale(total_loss).backward()

            # Unscale gradients before optimizer step (important for stability)
            grad_scaler.unscale_(optimizer)

            # Optional: Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer using scaled gradients
            grad_scaler.step(optimizer)

            # Update the GradScaler for the next iteration
            grad_scaler.update()


        train_dice_losses.append(dice_loss_value.item())
        train_cons_losses.append(cons_loss.item())
        train_lrs.append(optimizer.param_groups[0]["lr"])

        # Update metrics - change variable name here too
        metric_logger.update(dice_loss=dice_loss_value.item(), cons_loss=cons_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # visual_dict = None
        if visdir is not None and cur_iteration % visual_freq==0:
        # if visdir is not None and cur_iteration % visual_freq == 0:
            visual_dict['Logits_Augmented'] = torch.argmax(logits_aug,1).cpu().numpy()[0]
            visual_dict['ClassMixedLabel'] = torch.argmax(mixed_lbl,1).cpu().numpy()[0]
            
            fs = int(len(visual_dict) ** 0.5) + 1
            plt.figure(figsize=(fs * 4, fs * 4))
            for idx, k in enumerate(visual_dict.keys()):
                # print(k)
                plt.subplot(fs, fs, idx + 1)
                plt.title(k, fontsize=12, fontweight='bold')
                plt.axis('off')
                if k not in ['GT','GLA_pred','SBF_pred','classmixed_GT']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png', dpi=300, bbox_inches='tight')
            plt.close()

        cur_iteration += 1
        if cur_iteration >= max_iteration and max_iteration > 0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return cur_iteration

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    def convert_to_one_hot(tensor,num_c):
        return F.one_hot(tensor,num_c).permute((0,3,1,2))
    dices=[]
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)
        num_classes=logits.size(1)
        pred=torch.argmax(logits,dim=1)
        one_hot_pred=convert_to_one_hot(pred,num_classes)
        one_hot_gt=convert_to_one_hot(lbl,num_classes)
        dice=compute_meandice(one_hot_pred,one_hot_gt,include_background=False)
        dices.append(dice.cpu().numpy())
    dices=np.concatenate(dices,0)
    dices=np.nanmean(dices,0)
    return dices

def prediction_wrapper(model, test_loader, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    model.eval()
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()

            pred = model(img)
            pred=torch.argmax(pred,1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_meandice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names

