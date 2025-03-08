from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
# import bezier
import numpy as np

class HybridAugmentor(nn.Module):
    def __init__(self, num_classes, tau_max=0.7, tau_min=0.3, gamma=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.gamma = gamma
        self.controller = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )  # Progressive augmentation controller

    def class_aware_mixup(self, x, masks, alpha=0.4):
        """Class-aware nonlinear mixup augmentation"""
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)
        
        # Generate mixed image using class-specific masks
        perm = torch.randperm(batch_size)
        mixed_x = torch.zeros_like(x)
        for c in range(self.num_classes):
            mask = masks[:,c].unsqueeze(1)
            mixed_x += mask * (lam * x + (1-lam) * x[perm]) + (1-mask) * x
            
        return mixed_x, lam
    
    def bezier_curve(control_points, num_points=100):
        """Compute Bézier curve using De Casteljau’s algorithm."""
        def de_casteljau(t, points):
            while len(points) > 1:
                points = [(1 - t) * p + t * q for p, q in zip(points[:-1], points[1:])]
            return points[0]

        t_values = np.linspace(0, 1, num_points)
        curve = np.array([de_casteljau(t, control_points) for t in t_values])
        return curve

    def bezier_transform(self, x, masks, control_points=3):
        """Class-specific Bézier curve transformation."""
        B, C, H, W = x.shape
        transformed = torch.zeros_like(x)

        for c in range(self.num_classes):
            mask = masks[:, c].unsqueeze(1)

            # Random control points for Bézier
            nodes = np.random.uniform(0, 1, (control_points, 2))
            bezier_points = bezier_curve(nodes, num_points=H * W)

            sampled_x = bezier_points[:, 0].reshape(H, W)
            sampled_y = bezier_points[:, 1].reshape(H, W)

            # Apply affine transformation using the Bézier outputs
            warped = TF.affine(x, angle=0, translate=[0, 0], scale=1, shear=sampled_x)
            warped = TF.affine(warped, angle=0, translate=[0, 0], scale=1, shear=sampled_y)

            transformed += mask * warped

        return transformed

    def adaptive_threshold(self, epoch, total_epochs):
            """Sigmoidal curriculum learning for saliency threshold"""
            t = epoch / total_epochs
            return self.tau_min + (self.tau_max - self.tau_min) * (2/(1 + np.exp(-self.gamma*t)) - 1)

    def forward(self, x, masks, epoch, total_epochs):
            # Phase 1: Class-aware nonlinear mixup
            mixed_x, lam = self.class_aware_mixup(x, masks)
            
            # Phase 2: Location-scale transformation
            global_aug = self.bezier_transform(mixed_x, masks)
            local_aug = self.bezier_transform(x, masks)
            
            # Controller-adjusted parameters
            control_params = self.controller(torch.tensor([lam, epoch/total_epochs, 
                                                        global_aug.mean(), local_aug.std()]))
            alpha_ctrl, beta_ctrl, gamma_ctrl = torch.sigmoid(control_params)
            
            # Adaptive saliency fusion
            with torch.enable_grad():
                grad_global = torch.autograd.grad(global_aug.sum(), global_aug, create_graph=True)[0]
            saliency = (grad_global.abs().mean(1, keepdim=True) > 
                    self.adaptive_threshold(epoch, total_epochs)).float()
            
            fused = saliency * global_aug + (1 - saliency) * local_aug
            return fused

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
    aug_module = HybridAugmentor(num_classes=5)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
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
            augmented = aug_module(img, lbl, cur_iteration, warmup_iteration)
            # Forward passes
            logits_orig, feats_orig = model(img, return_features=True)
            logits_aug, feats_aug = model(augmented, return_features=True)
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
    aug_module = HybridAugmentor(num_classes=5)

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
        augmented = aug_module(img, lbl, cur_iteration, max_iteration)

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



def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,config=None,visdir=None):
    model.train()
    criterion.train()
    aux_criterion = SemanticConsistencyLoss()
    aug_module = HybridAugmentor(num_classes=5)

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
        grad_scaler = None
        # Generate augmented sample
        augmented = aug_module(img, lbl, cur_iteration, max_iteration)

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

