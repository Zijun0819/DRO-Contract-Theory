import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='aigc_data.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results\\', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 这行代码的作用是启用cuDNN自动调优器（autotuner）。这个调优器会在运行时为当前的硬件配置寻找最优的算法来执行操作，这通常会提高程序运行的速度
    torch.backends.cudnn.benchmark = True

    # data loading
    print(f"=> using dataset {config.data.train_dataset}_{config.data.data_volume}")
    DATASET = datasets.__dict__[config.data.type](config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    if config.training.t_method == "FL":
        diffusion.fl_train(DATASET)
    elif config.training.t_method == "Server":
        diffusion.server_train(DATASET)
    elif config.training.t_method == "SFL":
        diffusion.single_fl_train(DATASET)


if __name__ == "__main__":
    main()
