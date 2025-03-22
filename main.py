import argparse, os, sys, datetime, importlib
os.environ['KMP_DUPLICATE_LIB_OK']='true'
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from engine import train_warm_up,evaluate,train_one_epoch_SBF,train_one_epoch,prediction_wrapper
from losses import SetCriterion
import numpy as np
import random
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F # Import the necessary module
from globals import train_dice_losses, train_cons_losses, train_lrs, val_dice_losses, val_epochs
import types

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

import matplotlib.pyplot as plt

def plot_training_curves(train_dice_losses, train_cons_losses, train_lrs, val_dice_losses, val_epochs):
    """
    Plots training Dice Loss, Consistency Loss, and Learning Rate over iterations.
    Also plots validation Dice Loss at specified validation epochs.

    Args:
        train_dice_losses (list): List of training Dice Loss values.
        train_cons_losses (list): List of training Consistency Loss values.
        train_lrs (list): List of recorded learning rates.
        val_dice_losses (list): List of validation Dice Loss values.
        val_epochs (list): List of epochs where validation was performed.
    """
    
    # Plot Training Dice Loss with Validation Dice Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_dice_losses)), train_dice_losses, label="Train Dice Loss", color='blue')
    
    if val_dice_losses and val_epochs:
        plt.scatter(val_epochs, val_dice_losses, label="Validation Dice Loss", color='red', marker='o')
    
    plt.xlabel("Iterations")
    plt.ylabel("Dice Loss")
    plt.legend()
    plt.title("Dice Loss During Training and Validation")
    plt.show()

    # Plot Consistency Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_cons_losses)), train_cons_losses, label="Consistency Loss", color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Consistency Loss During Training")
    plt.show()

    # Plot Learning Rate
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_lrs)), train_lrs, label="Learning Rate", color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("LR")
    plt.legend()
    plt.title("Learning Rate Schedule")
    plt.show()


class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size,max_epoch, train=None, validation=None, test=None,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed=seed_everything(opt.seed)

    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name=None
        raise ValueError('no config')

    nowname = now +f'_seed{seed}'+ name + opt.postfix
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir= os.path.join(logdir, "visuals")
    for d in [logdir, cfgdir, ckptdir,visdir ]:
        os.makedirs(d, exist_ok=True)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))

    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())

    SBF_config = config.pop('saliency_balancing_fusion',OmegaConf.create())

    model = instantiate_from_config(model_config)
    # model = UNetWithResNet50Encoder(num_classes=5, pretrained=True)
    if torch.cuda.is_available():
        model=model.cuda()
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        else:
            checkpoint = torch.load(opt.resume, map_location="cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {opt.resume}")
    # Store original forward method
    original_forward = model.forward

    # Define new forward method to return both outputs and feature maps
    def new_forward(self, x, return_features=False):
        features = self.encoder(x)  # Extract features from encoder
        decoder_output = self.decoder(*features)  # Decode features
        masks = self.segmentation_head(decoder_output)  # Get segmentation mask
        
        if return_features:
            return masks, features
        return masks

    # Patch the model with the new forward method
    model.forward = types.MethodType(new_forward, model)

    if getattr(model_config.params, 'base_learning_rate') :
        bs, base_lr = config.data.params.batch_size, optimizer_config.base_learning_rate
        lr = bs * base_lr
    else:
        bs, lr = config.data.params.batch_size, optimizer_config.learning_rate

    if getattr(model_config.params, 'pretrain') :
        param_dicts = model.optim_parameters()
    else:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]
    # Ensure all layers are trainable (Fine-tuning entire model)
    for param in model.parameters():
        param.requires_grad = True

    opt_params = {'lr': lr}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
    opt = eval(optimizer_config['target'])(param_dicts, **opt_params)

    if optimizer_config.lr_scheduler =='lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch-50 + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError

    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch=999
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch= optimizer_config.max_epoch

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets["train"]))
    train_loader=DataLoader(data.datasets["train"], batch_size=data.batch_size,
                          num_workers=data.num_workers, shuffle=True, persistent_workers=True, drop_last=True, pin_memory = True)

    val_loader=DataLoader(data.datasets["validation"], batch_size=data.batch_size,  num_workers=1)

    if data.datasets.get('test') is not None:
        test_loader=DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase=True
    else:
        test_phase=False

    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter>0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)
    cur_iter=0
    best_dice=0
    label_name=data.datasets["train"].all_label_names
    for cur_epoch in range(max_epoch):
        if SBF_config.usage:
            # Set the current epoch for curriculum learning
            data.datasets["train"].set_epoch(cur_epoch)
            cur_iter = train_one_epoch_SBF(model, criterion,train_loader,opt,torch.device('cuda'),cur_epoch, cur_iter, max_epoch, optimizer_config.max_iter, SBF_config, visdir )
        else:
            cur_iter = train_one_epoch(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        if scheduler is not None:
            scheduler.step()

        # Early Stopping Parameters
        patience = 5  # Stop training if no improvement for `patience` validations
        patience_counter = 0
        best_dice = 0
        # Save Best Model on Validation (Every 100 Epochs)
        if (cur_epoch + 1) % 2 == 0:
            cur_dice = evaluate(model, val_loader, torch.device('cuda'))  # Compute validation score
            mean_dice = np.mean(cur_dice)

            if mean_dice > best_dice:
                best_dice = mean_dice
                patience_counter = 0  # Reset patience
                
                # Remove old validation checkpoints
                for f in os.listdir(ckptdir):
                    if 'val' in f:
                        os.remove(os.path.join(ckptdir, f))
                        
                # Save new best model
                torch.save({'model': model.state_dict()}, os.path.join(ckptdir, f'val_best_epoch_{cur_epoch}.pth'))
            
            else:
                patience_counter += 1  # No improvement, increment patience counter
            
            val_dice_losses.append(mean_dice)
            val_epochs.append(cur_epoch)
            # Print validation results
            str_out = f'Epoch [{cur_epoch}]   '
            for i, d in enumerate(cur_dice):
                str_out += f'Class {i}: {d}, '
            str_out += f'Validation DICE {mean_dice}/{best_dice}'
            print(str_out)
            
            # Early Stopping Check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {cur_epoch+1} epochs!")
                break  # Stop training

        # Save Latest Model Every Epochs
        if (cur_epoch + 1) % 5 == 0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, f'_epoch_{cur_epoch}_.pth'))

        # Stop Training if Max Iterations Reached
        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter > 0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break
    
    plot_training_curves(train_dice_losses, train_cons_losses, train_lrs, val_dice_losses, val_epochs)
