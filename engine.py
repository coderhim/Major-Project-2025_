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
from monai.losses import DiceLoss
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
from globals import train_dice_losses, train_cons_losses, train_lrs
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
# import bezier
import numpy as np
dice_loss = DiceLoss(to_onehot_y=False,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)
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

    # def class_aware_mixup(self, x, masks, alpha=0.4):
    #     """Class-aware nonlinear mixup augmentation"""
    #     batch_size = x.size(0)
    #     lam = np.random.beta(alpha, alpha)
        
    #     # Generate mixed image using class-specific masks
    #     perm = torch.randperm(batch_size)
    #     mixed_x = torch.zeros_like(x)
    #     for c in range(self.num_classes):
    #         mask = masks[:,c].unsqueeze(1)
    #         print("Shape of X is :",x.shape)
    #         print("shape of mask  is : ", mask.shape)
    #         mixed_x += mask * (lam * x + (1-lam) * x[perm]) + (1-mask) * x
            
    #     return mixed_x, lam
    
    def class_aware_mixup(self, x, masks, alpha=0.4):
        """
        Class-aware Mixup for multi-class foreground and background handling.

        Parameters:
        - x: Input tensor [batch_size, 1, 192, 192]
        - masks: Class mask [batch_size, 5, 192, 192] (0 for background)
        - alpha: Beta distribution parameter for λ

        Returns:
        - mixed_x: Mixed image
        - lam: Mixing coefficient
        """
        batch_size, _, height, width = x.shape
        num_classes = masks.shape[1]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Shuffle for mixup
        perm = torch.randperm(batch_size)

        # Ensure mask size matches input
        masks = masks[:, :, :height, :width]

        # Initialize output
        mixed_x = torch.zeros_like(x)

        # Handle background (mask == 0)
        background_mask = (masks.sum(dim=1, keepdim=True) == 0)
        mixed_x += background_mask * x

        # Iterate through each foreground class (mask > 0)
        for c in range(1, num_classes + 1):
            # Identify pixels belonging to class 'c'
            class_mask = (masks[:, c - 1] == 1).unsqueeze(1)  # Shape: [32, 1, 192, 192]
            # print("Shape of class mask: ",class_mask.shape)
            # print("Shape of x : ",x.shape)
            # Skip if no pixels for this class
            if class_mask.sum() == 0:
                continue

            # Apply class-aware mixup
            mixed_x += class_mask * (lam * x + (1 - lam) * x[perm])

        return mixed_x, lam
        
    def bezier_transform(self, x, masks, control_points=3):
        """
        Class-specific Bézier curve transformation without using the bezier library.
        Uses grid sampling to achieve per-pixel deformations similar to the original.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            masks: Class-specific masks of shape [B, num_classes, H, W]
            control_points: Number of control points for the Bézier curve
        
        Returns:
            Transformed tensor of the same shape as input
        """
        import torch.nn.functional as F
        
        B, C, H, W = x.shape
        transformed = torch.zeros_like(x)
        
        # Bernstein polynomial basis function
        def bernstein_poly(i, n, t):
            """
            Bernstein polynomial of degree n, term i.
            
            Args:
                i: Index of the term
                n: Degree of polynomial
                t: Parameter value (0 to 1)
                
            Returns:
                Value of the Bernstein polynomial at t
            """
            from math import comb
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        
        # Evaluate Bezier curve for a specific t value
        def bezier_curve_point(t, control_points_array):
            """
            Calculate point on Bezier curve for parameter t.
            
            Args:
                t: Parameter value (0 to 1)
                control_points_array: Control points array of shape [2, n_points]
                
            Returns:
                Point coordinates [x, y]
            """
            n = control_points_array.shape[1] - 1
            point = np.zeros(2)
            
            for i in range(n + 1):
                coef = bernstein_poly(i, n, t)
                point[0] += control_points_array[0, i] * coef
                point[1] += control_points_array[1, i] * coef
                
            return point
        
        # Create normalized sampling grid (from -1 to 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=2).to(x.device)
        
        for c in range(self.num_classes):
            mask = masks[:, c].unsqueeze(1)
            
            # Generate random control points
            nodes = np.random.uniform(0, 1, (2, control_points))
            
            # Create t values corresponding to flattened image pixels
            t_values = torch.linspace(0, 1, H*W)
            
            # Evaluate Bezier curve for each t value (using numpy for simplicity)
            curve_points = np.zeros((H*W, 2))
            for i, t in enumerate(t_values.numpy()):
                curve_points[i] = bezier_curve_point(t, nodes)
            
            # Reshape to match image dimensions
            curve_points = curve_points.reshape(H, W, 2)
            
            # Convert to displacement field (scaled to reasonable values)
            # Normalize to range -0.2 to 0.2 for displacement
            displacement_field = torch.from_numpy(curve_points).float().to(x.device) - 0.5
            displacement_field = displacement_field * 0.4  # Scale factor to control deformation strength
            
            # Apply the displacement to base grid
            sampling_grid = base_grid.clone().repeat(B, 1, 1, 1)
            sampling_grid += displacement_field.unsqueeze(0).repeat(B, 1, 1, 1)
            
            # Use grid_sample to apply the deformation
            warped = F.grid_sample(
                x, 
                sampling_grid,
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
            
            transformed += mask * warped
        
        return transformed
    
    def to(self, device):
        self.controller = self.controller.to(device)
        return self
    def adaptive_threshold(self, epoch, total_epochs):
            """Sigmoidal curriculum learning for saliency threshold"""
            t = epoch / total_epochs
            return self.tau_min + (self.tau_max - self.tau_min) * (2/(1 + np.exp(-self.gamma*t)) - 1)

    def forward(self, x, masks, epoch, total_epochs):
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = x.device
            # Phase 1: Class-aware nonlinear mixup
            mixed_x, lam = self.class_aware_mixup(x, masks)
            
            # Phase 2: Location-scale transformation
            global_aug = self.bezier_transform(mixed_x, masks)
            local_aug = self.bezier_transform(x, masks)
            
            ## Ensure 'global_aug' retains gradients
            global_aug.requires_grad_(True)

            # Controller-adjusted parameters (keep in computational graph)
            control_params = self.controller(torch.stack([
                torch.tensor(lam, device=device),  # Convert lam to a tensor
                torch.tensor(epoch / total_epochs, device=device),
                global_aug.mean(),
                local_aug.std()
            ]))


            alpha_ctrl, beta_ctrl, gamma_ctrl = torch.sigmoid(control_params)

            # Compute saliency map without detaching the graph
            grad_global = torch.autograd.grad(global_aug.sum(), global_aug, create_graph=True, retain_graph=True)[0]

            # Apply adaptive saliency threshold
            saliency = (grad_global.abs().mean(1, keepdim=True) > self.adaptive_threshold(epoch, total_epochs)).float()

            # Fused output
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
    
    global train_lrs, train_cons_losses, train_dice_losses

    model.train()
    criterion.train()
    aux_criterion = SemanticConsistencyLoss()
    aug_module = HybridAugmentor(num_classes=5).to(device)  # Move to the same device

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 0
    visual_freq = 100  # Added visual frequency
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move samples to the appropriate device
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']
        grad_scaler = 1
        
        # Generate augmented sample
        lbl = F.one_hot(lbl,5).permute((0,3,1,2))
        augmented = aug_module(img, lbl, cur_iteration, max_iteration)

        if grad_scaler is None:
            # Regular precision
            # logits_orig, feats_orig = model(img, return_features=True)
            # logits_aug, feats_aug = model(augmented, return_features=True)
            logits_orig = model(img)
            feats_orig_list = model.encoder(img)
            # feats_orig = feats_orig_list[-1] #last feature map
            logits_aug = model(augmented)
            feats_aug_list = model.encoder(augmented)
            # feats_aug = feats_aug_list[-1]

            # Note the change here - using the global dice_loss function but storing result in dice_loss_value
            # print("model Output Shape : ", logits_orig.shape)
            # print("model Output augmented Shape : ", logits_aug.shape)
            # print("label Shape : ", lbl.shape)
            # print("features Shape : ", feats_orig.shape)
            # print("features augmented Shape : ", feats_aug.shape)
            dice_loss_value = (dice_loss(logits_orig, lbl) + dice_loss(logits_aug, lbl) ) /2
            cons_loss = aux_criterion(feats_orig_list, feats_aug_list)
            total_loss = dice_loss_value + cons_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            # Mixed precision (AMP)
            # Ensure GradScaler is initialized before training starts
            # if "grad_scaler" not in globals():
            grad_scaler = torch.amp.GradScaler()

            # Mixed precision (AMP) training
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_orig = model(img)
                feats_orig_list = model.encoder(img)
                logits_aug = model(augmented)
                feats_aug_list = model.encoder(augmented)

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

        visual_dict = None
        if cur_iteration % visual_freq == 0:
            visual_dict = {}
            visual_dict['Original'] = img.detach().cpu().numpy()[0, 0]
            visual_dict['Augmented'] = augmented.detach().cpu().numpy()[0, 0]
            visual_dict['GT'] = lbl.detach().cpu().numpy()[0]
        
        if visdir is not None and cur_iteration % visual_freq == 0:
            visual_dict['Logits_Original'] = torch.argmax(logits_orig,1).cpu().numpy()[0]
            visual_dict['Logits_Augmented'] = torch.argmax(logits_aug,1).cpu().numpy()[0]
            
            fs = int(len(visual_dict) ** 0.5) + 1
            plt.figure(figsize=(fs * 4, fs * 4))
            for idx, k in enumerate(visual_dict.keys()):
                print(k)
                plt.subplot(fs, fs, idx + 1)
                plt.title(k, fontsize=12, fontweight='bold')
                plt.axis('off')
                if k not in ['GT']:
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

