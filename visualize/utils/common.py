import random
import torch
import numpy as np
import time
import json
import os
from omegaconf import OmegaConf


def setup_deterministic(seed: int = 2024):
    """ 실험 재현성 확보를 위해서 random관련 함수들의 seed 설정 """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)