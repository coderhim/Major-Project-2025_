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

class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
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

import torch
import torch.nn as nn
import torchvision.models as models
class UNetWithResNet50Encoder(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(UNetWithResNet50Encoder, self).__init__()
        # Load pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept single-channel input
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.encoder.add_module('bn1', resnet.bn1)
        self.encoder.add_module('relu', resnet.relu)
        self.encoder.add_module('maxpool', resnet.maxpool)
        self.encoder.add_module('layer1', resnet.layer1)
        self.encoder.add_module('layer2', resnet.layer2)
        self.encoder.add_module('layer3', resnet.layer3)
        self.encoder.add_module('layer4', resnet.layer4)
        
        # Decoder
        self.upconv4 = self._upconv(2048, 1024)
        self.upconv3 = self._upconv(1024, 512)
        self.upconv2 = self._upconv(512, 256)
        self.upconv1 = self._upconv(256, 64)
        
        # Add an extra upsampling layer to account for the initial stride=2 and maxpool
        self.upconv0 = self._upconv(64, 64)
        self.upconv_final = self._upconv(64, 64)  # For final upsampling to match input size
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_features=False):
        input_size = x.size()  # Store original input size
        
        # Encoder
        conv1 = self.encoder.conv1(x)        # Downsampling 1 (stride=2)
        bn1 = self.encoder.bn1(conv1)
        relu = self.encoder.relu(bn1)
        maxpool = self.encoder.maxpool(relu)  # Downsampling 2 (stride=2)
        layer1 = self.encoder.layer1(maxpool)
        layer2 = self.encoder.layer2(layer1)  # Downsampling 3 (stride=2)
        layer3 = self.encoder.layer3(layer2)  # Downsampling 4 (stride=2)
        layer4 = self.encoder.layer4(layer3)  # Downsampling 5 (stride=2)
        
        # Decoder with skip connections
        up4 = self.upconv4(layer4)  # Upsampling 1
        up3 = self.upconv3(up4 + layer3)  # Upsampling 2
        up2 = self.upconv2(up3 + layer2)  # Upsampling 3
        up1 = self.upconv1(up2 + layer1)  # Upsampling 4
        up0 = self.upconv0(up1)  # Upsampling 5 (to account for maxpool)
        
        # Add an extra upsampling to match original input size
        up_final = self.upconv_final(up0)  # Upsampling 6 (to match original input size)
        
        # Apply final 1x1 convolution to get output classes
        logits = self.final_conv(up_final)
        
        # Ensure output size matches input size
        if logits.size()[2:] != input_size[2:]:
            logits = F.interpolate(logits, size=input_size[2:], mode='bilinear', align_corners=True)
        
        if return_features:
            return logits, layer4
        else:
            return logits
# class UNetWithResNet50Encoder(nn.Module):
#     def __init__(self, num_classes, pretrained=True):
#         super(UNetWithResNet50Encoder, self).__init__()
#         # Load pre-trained ResNet50 model
#         resnet = models.resnet50(pretrained=pretrained)
        
#         # Modify the first convolutional layer to accept single-channel input
#         self.encoder = nn.Sequential()
#         self.encoder.add_module('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
#         self.encoder.add_module('bn1', resnet.bn1)
#         self.encoder.add_module('relu', resnet.relu)
#         self.encoder.add_module('maxpool', resnet.maxpool)
#         self.encoder.add_module('layer1', resnet.layer1)
#         self.encoder.add_module('layer2', resnet.layer2)
#         self.encoder.add_module('layer3', resnet.layer3)
#         self.encoder.add_module('layer4', resnet.layer4)
        
#         # Decoder
#         self.upconv4 = self._upconv(2048, 1024)
#         self.upconv3 = self._upconv(1024, 512)
#         self.upconv2 = self._upconv(512, 256)
#         self.upconv1 = self._upconv(256, 64)
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
#     def _upconv(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x, return_features=False):
#         # Encoder
#         conv1 = self.encoder.conv1(x)
#         bn1 = self.encoder.bn1(conv1)
#         relu = self.encoder.relu(bn1)
#         maxpool = self.encoder.maxpool(relu)
#         layer1 = self.encoder.layer1(maxpool)
#         layer2 = self.encoder.layer2(layer1)
#         layer3 = self.encoder.layer3(layer2)
#         layer4 = self.encoder.layer4(layer3)
        
#         # Decoder
#         up4 = self.upconv4(layer4)
#         up3 = self.upconv3(up4 + layer3)
#         up2 = self.upconv2(up3 + layer2)
#         up1 = self.upconv1(up2 + layer1)
#         logits = self.final_conv(up1 + relu)
        
#         if return_features:
#             return logits, layer4
#         else:
#             return logits

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed=seed_everything(opt.seed)
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
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

    # model = instantiate_from_config(model_config)
    model = UNetWithResNet50Encoder(num_classes=5, pretrained=True)
    if torch.cuda.is_available():
        model=model.cuda()

    if getattr(model_config.params, 'base_learning_rate') :
        bs, base_lr = config.data.params.batch_size, optimizer_config.base_learning_rate
        lr = bs * base_lr
    else:
        bs, lr = config.data.params.batch_size, optimizer_config.learning_rate

    if getattr(model_config.params, 'pretrain') :
        param_dicts = model.optim_parameters()
    else:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]

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
            cur_iter = train_one_epoch_SBF(model, criterion,train_loader,opt,torch.device('cuda'),cur_epoch,cur_iter, optimizer_config.max_iter, SBF_config, visdir)
        else:
            cur_iter = train_one_epoch(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        if scheduler is not None:
            scheduler.step()

        # Save Bset model on val
        if (cur_epoch+1)%100==0:
            cur_dice = evaluate(model, val_loader, torch.device('cuda'))
            if np.mean(cur_dice)>best_dice:
                best_dice=np.mean(cur_dice)
                for f in os.listdir(ckptdir):
                    if 'val' in f:
                        os.remove(os.path.join(ckptdir,f))
                torch.save({'model': model.state_dict()}, os.path.join(ckptdir,f'val_best_epoch_{cur_epoch}.pth'))

            str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(cur_dice):
                str+=f'Class {i}: {d}, '
            str+=f'Validation DICE {np.mean(cur_dice)}/{best_dice}'
            print(str)

        # Save latest model
        if (cur_epoch+1)%50==0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir,'latest.pth'))

        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter>0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break

