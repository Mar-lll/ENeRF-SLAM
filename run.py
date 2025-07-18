import argparse
import random

import numpy as np
import torch

from src import config
from src.ENeRF_SLAM import ENeRF_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the ENeRF-SLAM'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.set_defaults(nice=False)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/enerf_slam.yaml')

    slam = ENeRF_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
